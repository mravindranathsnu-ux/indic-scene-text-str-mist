"""
vocab.py
Character-level vocabulary for scene text recognition, generalized across
scripts. Originally built for Telugu; `TeluguVocab` is kept as a backward-
compatible alias for `Vocab` so existing telugu_vocab.json files and code
still work unchanged.

Supports building a vocab three ways:
  - Vocab.build_from_labels(labels)                 : from your own data
  - Vocab.build_from_unicode_block(language="hindi") : full script coverage
  - Vocab.load(path)                                  : a saved/custom vocab
"""
import json

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Common punctuation/digits shared across scripts. Native-script digits
# (Devanagari ०-९, Tamil ௦-௯, etc.) are already included in each script's
# Unicode block range below, so this is only the ASCII/Latin extras.
_COMMON_EXTRAS = list(" 0123456789.,:;!?()-'\"")

# Unicode block ranges for major Indian scripts + English/Latin. Each entry
# is a list of (start, end) inclusive codepoint ranges. Add a new script by
# adding an entry here — everything else in this file is script-agnostic.
UNICODE_BLOCKS = {
    "telugu":     [(0x0C00, 0x0C7F)],
    "hindi":      [(0x0900, 0x097F)],   # Devanagari — also covers Marathi, Sanskrit, Nepali, etc.
    "devanagari": [(0x0900, 0x097F)],   # alias
    "marathi":    [(0x0900, 0x097F)],   # alias — Devanagari; your own vocab file may add ASCII digits/punct
    "tamil":      [(0x0B80, 0x0BFF)],
    "malayalam":  [(0x0D00, 0x0D7F)],
    "kannada":    [(0x0C80, 0x0CFF)],
    "bengali":    [(0x0980, 0x09FF)],
    "assamese":   [(0x0980, 0x09FF)],   # alias — Bengali-Assamese script (adds ৰ, ৱ within this block)
    "gujarati":   [(0x0A80, 0x0AFF)],
    "gurmukhi":   [(0x0A00, 0x0A7F)],   # Punjabi
    "punjabi":    [(0x0A00, 0x0A7F)],   # alias
    "odia":       [(0x0B00, 0x0B7F)],
    "meitei":     [(0xABC0, 0xABFF)],   # Meetei Mayek script (Manipuri is also often written in Bengali script — see note below)
    "manipuri":   [(0xABC0, 0xABFF)],   # alias
    "urdu":       [(0x0600, 0x06FF)],   # Arabic block (Urdu is RTL — see note in build_from_unicode_block below)
    "english":    [(0x0041, 0x005A), (0x0061, 0x007A)],  # A-Z, a-z (digits/punct come from _COMMON_EXTRAS)
    "latin":      [(0x0041, 0x005A), (0x0061, 0x007A)],  # alias
}
# Note: Manipuri (Meitei) is written in both Meetei Mayek and Bengali-Assamese
# script depending on context — a real-world vocab for it (e.g. one you
# already have from another OCR project) may mix characters from both
# blocks. build_from_unicode_block("meitei") only covers the Meetei Mayek
# block; bring your own vocab file (Vocab.load()) for full real-world
# coverage if your data uses both scripts.


class Vocab:
    def __init__(self, chars=None):
        self.chars = chars or []
        self._build()

    def _build(self):
        tokens = SPECIAL_TOKENS + sorted(set(self.chars))
        self.stoi = {ch: i for i, ch in enumerate(tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.sos_id = self.stoi[SOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    @classmethod
    def build_from_labels(cls, labels):
        """labels: iterable of ground-truth strings, any script."""
        charset = set()
        for label in labels:
            charset.update(list(label))
        return cls(chars=list(charset))

    @classmethod
    def build_from_unicode_block(cls, language="telugu"):
        """
        Fallback vocab covering a full script's Unicode block plus common
        punctuation/digits. Useful before you have a labeled dataset to
        derive an exact vocab from. `language` must be a key in
        UNICODE_BLOCKS (case-insensitive) — see that dict for supported
        scripts (major Indian scripts + English/Latin).
        """
        key = language.lower()
        if key not in UNICODE_BLOCKS:
            raise ValueError(
                f"Unknown language '{language}'. Supported: {sorted(set(UNICODE_BLOCKS))}. "
                f"To add a new script, add its Unicode block range to UNICODE_BLOCKS in vocab.py."
            )
        chars = []
        for start, end in UNICODE_BLOCKS[key]:
            chars.extend(chr(c) for c in range(start, end + 1))
        chars += _COMMON_EXTRAS
        return cls(chars=chars)

    def encode(self, text, add_special_tokens=True):
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, strip_special=True):
        chars = []
        for i in ids:
            ch = self.itos.get(int(i), UNK_TOKEN)
            if strip_special and ch in SPECIAL_TOKENS:
                if ch == EOS_TOKEN:
                    break
                continue
            chars.append(ch)
        return "".join(chars)

    def __len__(self):
        return len(self.stoi)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            chars = json.load(f)
        return cls(chars=chars)


# Backward-compatible alias — existing code/imports using TeluguVocab keep working.
TeluguVocab = Vocab
