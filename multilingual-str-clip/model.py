"""
model.py
CLIP-backbone encoder + Transformer decoder for scene text recognition —
script/language-agnostic (Telugu, other Indian scripts, or English; see
vocab.py's UNICODE_BLOCKS for supported languages). One model = one
script/vocab; train separate models per language rather than one combined
multilingual model, to avoid diluting decoder capacity and vocab size
across scripts with limited per-language training data.

Transfer-learning strategy:
  - Load a pretrained CLIP visual encoder (ViT) as the image feature extractor.
  - Freeze most of the backbone; optionally unfreeze the last N transformer
    blocks for fine-tuning on your target script's scene-text data.
  - Feed the CLIP patch-token sequence (not just the pooled CLS embedding)
    as "memory" into a standard Transformer decoder that autoregressively
    generates character sequences. Patch tokens are used (instead of
    the single pooled vector CLIP normally outputs) because recognition
    needs spatial detail CLIP's pooled embedding throws away.
"""
import math
import torch
import torch.nn as nn

from clip_vit import load_clip_visual


class CLIPPatchEncoder(nn.Module):
    """
    Wraps a locally-loaded CLIP ViT visual backbone (see clip_vit.py — no
    `clip` pip package involved) and exposes the *sequence* of patch-token
    embeddings (before CLS pooling).
    """

    def __init__(self, clip_checkpoint_path, freeze_backbone=True, unfreeze_last_n=2, device="cpu"):
        super().__init__()
        self.visual, self.preprocess = load_clip_visual(clip_checkpoint_path, device=device)
        self.dtype = torch.float32

        if freeze_backbone:
            self.freeze_all()
            self.unfreeze_last_n_blocks(unfreeze_last_n)

        self.output_dim = self.visual.transformer.width

    def freeze_all(self):
        """Freeze the entire CLIP visual backbone."""
        for p in self.visual.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze only the last `n` transformer blocks (+ final layernorm),
        re-freezing everything else first. Call this mid-training to move
        from a "backbone fully frozen" warmup stage to a "lightly fine-tune
        the top of the backbone" stage — the standard recipe for adapting a
        pretrained vision backbone to a new domain/script without wrecking
        its pretrained low-level features.
        """
        self.freeze_all()
        if n > 0:
            blocks = self.visual.transformer.resblocks
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.visual.ln_post.parameters():
                p.requires_grad = True

    def forward(self, images):
        """
        images: (B, 3, H, W) already CLIP-preprocessed
        returns: (B, num_patches + 1, width) sequence of token embeddings
        """
        x = images.type(self.dtype)
        return self.visual.forward_features(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CLIPSceneTextOCR(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        clip_checkpoint_path,
        decoder_dim=512,
        decoder_heads=8,
        decoder_layers=4,
        decoder_ff=2048,
        max_len=32,
        freeze_backbone=True,
        unfreeze_last_n=2,
        dropout=0.1,
        device="cpu",
    ):
        super().__init__()
        self.encoder = CLIPPatchEncoder(clip_checkpoint_path, freeze_backbone, unfreeze_last_n, device=device)
        self.memory_proj = nn.Linear(self.encoder.output_dim, decoder_dim)

        self.token_embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(decoder_dim, max_len=max_len + 2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_proj = nn.Linear(decoder_dim, vocab_size)
        self.pad_id = pad_id

    @property
    def preprocess(self):
        return self.encoder.preprocess

    def unfreeze_backbone_last_n(self, n):
        """Move the CLIP backbone from its current freeze state to
        'only the last n transformer blocks trainable'. n=0 re-freezes
        everything (pure feature-extraction mode)."""
        self.encoder.unfreeze_last_n_blocks(n)

    @staticmethod
    def generate_causal_mask(size, device):
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def forward(self, images, tgt_token_ids):
        """
        images: (B, 3, H, W)
        tgt_token_ids: (B, T)  teacher-forcing decoder input (caller shifts right)
        returns logits: (B, T, vocab_size)
        """
        memory = self.encoder(images)                  # (B, S, enc_dim)
        memory = self.memory_proj(memory)               # (B, S, dec_dim)

        tgt_emb = self.token_embedding(tgt_token_ids)    # (B, T, dec_dim)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Both masks must share a dtype (PyTorch warns/deprecates mixing
        # bool + float). Rather than switch the causal mask to bool — an
        # untested code path for this project — convert the padding mask
        # to the same additive-float convention the causal mask already
        # uses, which keeps the causal masking numerically identical to
        # what was already verified to work.
        pad_bool = tgt_token_ids == self.pad_id
        tgt_key_padding_mask = torch.zeros_like(pad_bool, dtype=torch.float32)
        tgt_key_padding_mask = tgt_key_padding_mask.masked_fill(pad_bool, float("-inf"))
        causal_mask = self.generate_causal_mask(tgt_token_ids.size(1), tgt_token_ids.device)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(out)

    @torch.no_grad()
    def greedy_decode(self, images, vocab, max_len=32, device="cpu"):
        self.eval()
        memory = self.memory_proj(self.encoder(images))
        B = images.size(0)
        ys = torch.full((B, 1), vocab.sos_id, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            tgt_emb = self.pos_encoding(self.token_embedding(ys))
            causal_mask = self.generate_causal_mask(ys.size(1), device)
            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            logits = self.output_proj(out[:, -1, :])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= (next_tok.squeeze(1) == vocab.eos_id)
            if finished.all():
                break

        return [vocab.decode(seq.tolist()) for seq in ys]

    @torch.no_grad()
    def _beam_search_core(self, images, vocab, max_len, beam_size, length_penalty, device):
        """
        Shared beam search implementation used by both beam_search_decode
        (public, unchanged behavior: returns only the top candidate) and
        beam_search_decode_candidates (new: returns all beam_size
        candidates). Extracted verbatim from the original single-method
        beam_search_decode with NO logic changes, so the two public
        methods can never silently diverge from each other -- adding a
        second consumer of the beam search must not risk altering what
        beam_search_decode has always returned, since every existing
        evaluation result in this project depends on it.

        Returns (ys, norm_scores):
          ys:          (B, beam_size, T) token ids for every completed beam
          norm_scores: (B, beam_size) length-normalized scores, same beam order as ys
        """
        self.eval()
        B = images.size(0)
        memory = self.memory_proj(self.encoder(images))          # (B, S, D)
        S, D = memory.size(1), memory.size(2)
        vocab_size = self.output_proj.out_features

        # Expand memory across beams: (B, S, D) -> (B*beam, S, D)
        memory = memory.unsqueeze(1).expand(B, beam_size, S, D).reshape(B * beam_size, S, D)

        ys = torch.full((B * beam_size, 1), vocab.sos_id, dtype=torch.long, device=device)
        # Only the first beam of each batch item starts "active" so the
        # first expansion doesn't produce beam_size identical copies.
        scores = torch.full((B, beam_size), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.view(B * beam_size)
        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)
        # Effective length per beam: frozen once a beam finishes (its raw
        # score also stops changing at that point — see below), so both
        # stay self-consistent. Starts at 1 for the initial SOS token.
        lengths = torch.ones(B * beam_size, device=device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoding(self.token_embedding(ys))
            causal_mask = self.generate_causal_mask(ys.size(1), device)
            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            logits = self.output_proj(out[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)        # (B*beam, V)

            if finished.any():
                # Frozen beams may only "continue" via eos at zero extra
                # cost, so their raw score stops changing once finished.
                log_probs = log_probs.clone()
                log_probs[finished] = float("-inf")
                log_probs[finished, vocab.eos_id] = 0.0

            total_scores = (scores.unsqueeze(1) + log_probs).view(B, beam_size * vocab_size)

            # Already-finished beams keep their frozen length; still-active
            # beams grow by 1. Using each beam's OWN effective length (not
            # a value shared by every candidate at this step) is essential
            # — without this, a finished beam's frozen raw score wins every
            # later comparison purely because it stops accumulating cost,
            # permanently crowding out longer (possibly correct) candidates
            # regardless of --length_penalty. Verified against a standalone
            # toy simulation before shipping this fix.
            new_lengths = torch.where(finished, lengths, lengths + 1)
            new_lengths_exp = new_lengths.view(B, beam_size, 1).expand(B, beam_size, vocab_size)
            new_lengths_exp = new_lengths_exp.reshape(B, beam_size * vocab_size)

            norm_total_scores = total_scores / (new_lengths_exp ** length_penalty)
            _, topk_idx = norm_total_scores.topk(beam_size, dim=1)              # (B, beam)
            topk_scores = torch.gather(total_scores, 1, topk_idx)                # keep RAW scores
            topk_lengths = torch.gather(new_lengths_exp, 1, topk_idx)
            beam_idx = torch.div(topk_idx, vocab_size, rounding_mode="floor")
            token_idx = topk_idx % vocab_size

            batch_offset = (torch.arange(B, device=device) * beam_size).unsqueeze(1)
            flat_beam_idx = (beam_idx + batch_offset).view(-1)

            ys = torch.cat([ys[flat_beam_idx], token_idx.view(-1, 1)], dim=1)
            finished = finished[flat_beam_idx] | (token_idx.view(-1) == vocab.eos_id)
            scores = topk_scores.view(-1)
            lengths = topk_lengths.view(-1)

            if finished.all():
                break

        norm_scores = (scores / (lengths ** length_penalty)).view(B, beam_size)
        ys = ys.view(B, beam_size, -1)
        return ys, norm_scores

    @torch.no_grad()
    def beam_search_decode(self, images, vocab, max_len=32, beam_size=5, length_penalty=0.7, device="cpu"):
        """
        Batched beam search. Usually gives a small but real accuracy bump
        over greedy decoding (a few % WRR) since it can recover from a
        locally-plausible-but-wrong character choice that greedy decode
        would commit to irreversibly — useful precisely for cases like
        similar-looking characters (e.g. Indic conjuncts) where the top-1 logit at a given
        step is a close call.

        length_penalty: >0 penalizes shorter sequences less/more; 0.6-1.0
        is a typical range (0 = no length normalization).

        Behavior is UNCHANGED from before this method was refactored to
        share its core loop with beam_search_decode_candidates: this is
        still exactly "run the search, keep only the single best beam."
        """
        ys, norm_scores = self._beam_search_core(images, vocab, max_len, beam_size, length_penalty, device)
        best_beam = norm_scores.argmax(dim=1)
        B = ys.size(0)
        return [vocab.decode(ys[b, best_beam[b]].tolist()) for b in range(B)]

    @torch.no_grad()
    def beam_search_decode_candidates(self, images, vocab, max_len=32, beam_size=5, length_penalty=0.7, device="cpu"):
        """
        Same search as beam_search_decode, but returns ALL beam_size
        completed candidates per image (decoded text + normalized score),
        sorted best-first, instead of discarding every beam but the
        winner. Added specifically to support lexicon-aware reranking
        against real alternative hypotheses the model actually
        considered during decoding, rather than only the top-1 string.

        The search itself is identical to beam_search_decode (same
        shared core, see _beam_search_core) — only the final selection
        step differs.

        Returns: list of length B; each element is a list of beam_size
        (text, normalized_score) tuples, sorted by score descending
        (index 0 == what beam_search_decode would have returned).
        """
        ys, norm_scores = self._beam_search_core(images, vocab, max_len, beam_size, length_penalty, device)
        B = ys.size(0)
        results = []
        for b in range(B):
            order = torch.argsort(norm_scores[b], descending=True)
            candidates = [(vocab.decode(ys[b, i].tolist()), norm_scores[b, i].item()) for i in order]
            results.append(candidates)
        return results


# Backward-compatible alias — existing code/imports using CLIPTeluguOCR keep working.
CLIPTeluguOCR = CLIPSceneTextOCR
