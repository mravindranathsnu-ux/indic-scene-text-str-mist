"""
clip_vit.py
A minimal, self-contained CLIP Vision Transformer (ViT) — reimplemented
directly against plain torch/torchvision so this project has NO dependency
on the `clip` pip package. It reads the official OpenAI CLIP `.pt`
checkpoint files (e.g. ViT-B-32.pt, ViT-B-16.pt) exactly as released and
reconstructs only the visual tower (we never need CLIP's text encoder).

Official checkpoints are TorchScript archives, so loading is:
    1. torch.jit.load() to get all tensors back out as a state_dict
       (falls back to a plain torch.load() if the file is already a
       plain state dict).
    2. Infer the architecture's hyperparameters (width, depth, patch size,
       image resolution, projection dim) directly from tensor shapes in
       the state dict — the same trick the original CLIP repo uses.
    3. Build a plain nn.Module VisionTransformer with matching parameter
       names and load the weights into it.

No network access, no `clip`/`ftfy`/`regex` packages required — only
torch, torchvision, and Pillow.
"""
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
from PIL import Image

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        num_patches = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward_features(self, x):
        """Returns the full patch-token sequence (B, num_patches+1, width),
        i.e. everything up to (but not including) the final pooling and
        projection. This is what a recognition/detection head wants —
        it keeps spatial information the pooled embedding discards."""
        x = self.conv1(x)                                  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)           # (B, width, grid**2)
        x = x.permute(0, 2, 1)                              # (B, grid**2, width)
        cls_tok = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tok, x], dim=1)                  # (B, grid**2+1, width)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)      # NLD -> LND (nn.MultiheadAttention default is seq-first)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)      # LND -> NLD
        return x

    def forward(self, x):
        """Standard CLIP forward: pooled + projected embedding (CLS token
        only). Provided for completeness; the OCR model uses
        forward_features() instead."""
        x = self.forward_features(x)
        pooled = self.ln_post(x[:, 0, :])
        return pooled @ self.proj


def _load_raw_state_dict(checkpoint_path, device="cpu"):
    try:
        jit_model = torch.jit.load(checkpoint_path, map_location=device).eval()
        return jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict


def _build_vision_transformer(state_dict):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([
        k for k in state_dict.keys()
        if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
    ])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    output_dim = state_dict["visual.proj"].shape[1]
    heads = vision_width // 64  # true for all official CLIP ViT variants (B/32, B/16, L/14)

    model = VisionTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=heads,
        output_dim=output_dim,
    )

    visual_state = {
        k[len("visual."):]: v for k, v in state_dict.items() if k.startswith("visual.")
    }
    # ResidualAttentionBlock stores resblocks in an nn.ModuleList (not the
    # nn.Sequential the original repo uses), but attribute paths
    # ("transformer.resblocks.N....") are identical either way, so keys match.
    missing, unexpected = model.load_state_dict(visual_state, strict=False)
    if len(missing) > 0:
        raise RuntimeError(
            f"Checkpoint is missing expected visual-tower weights: {missing[:5]}... "
            f"Is this a valid OpenAI CLIP ViT checkpoint?"
        )
    return model, image_resolution


class LetterboxResize:
    """
    Resizes an image to fit *within* a `size` x `size` box while preserving
    its aspect ratio (no cropping, no stretching/distortion), then pads the
    leftover space with a neutral gray so the model still receives the
    fixed square input its positional embeddings expect.

    This is the right choice when your crops are already tight around the
    text (as opposed to full scene photos): the whole image is used, and
    unlike a plain squash-resize, characters keep their true proportions —
    important for Telugu, where vowel signs (matras) and conjuncts can
    become hard to distinguish if stretched vertically.
    """

    def __init__(self, size, fill=(127, 127, 127)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        resized = img.resize((new_w, new_h), Image.BICUBIC)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas


def build_preprocess(image_resolution):
    """
    IMPORTANT for OCR/scene-text: standard CLIP preprocessing is
    Resize(shorter side) -> CenterCrop(square), which is fine for natural
    photos but WRONG for text crops. Word/line images are typically wide
    and short (e.g. 150x40) — Resize(224) would scale the short side (40)
    up to 224, blowing the image out to ~840x224, and CenterCrop(224)
    would then keep only the center ~27% of that width, silently chopping
    off most of the word.

    Since crops here are already tight around the text (nothing to crop
    away), we use LetterboxResize instead: the whole image is scaled down
    to fit inside the square frame with its aspect ratio intact, then
    padded — no cropping, no distortion. (An earlier version of this
    function used a "squash" resize straight to image_resolution x
    image_resolution, which avoids cropping but still stretches the
    aspect ratio; letterboxing is strictly better for text where true
    glyph proportions matter.)

    A checkpoint trained under a different preprocessing scheme must be
    retrained after switching — the visual input distribution changes.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        LetterboxResize(image_resolution),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
    ])


def load_clip_visual(checkpoint_path, device="cpu"):
    """
    Loads only the visual (ViT) tower of an official OpenAI CLIP checkpoint
    from a local .pt file, with no `clip` package and no network access.

    Returns (model, preprocess) — model is a VisionTransformer in eval
    mode with float32 weights; preprocess is the matching torchvision
    transform (resize/crop/normalize) for that checkpoint's resolution.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"CLIP checkpoint not found at '{checkpoint_path}'. "
            f"Point --clip_ckpt at your local ViT-B-32.pt / ViT-B-16.pt file."
        )
    state_dict = _load_raw_state_dict(checkpoint_path, device="cpu")
    model, image_resolution = _build_vision_transformer(state_dict)
    model = model.float().to(device)
    preprocess = build_preprocess(image_resolution)
    return model, preprocess
