import json
from pathlib import Path
from typing import List, Union


class CharacterTokenizer:
    """
    Character-level tokenizer for ZaidGPT.
    Supports vocabulary persistence (save/load) and graceful handling of unknown tokens.
    """

    UNK_TOKEN = "<UNK>"

    def __init__(self, text: str = None, vocab_file: Union[str, Path] = None):
        self.stoi = {}
        self.itos = {}
        self.vocab = []

        if vocab_file and Path(vocab_file).exists():
            self.load(vocab_file)
        elif text is not None:
            self._build_vocab(text)
        else:
            # Default minimal ASCII vocabulary
            default_chars = " \n!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
            self._build_vocab(default_chars)

    def _build_vocab(self, text: str):
        # Unique characters sorted deterministically
        unique_chars = sorted(list(set(text)))
        if self.UNK_TOKEN not in unique_chars:
            unique_chars.insert(0, self.UNK_TOKEN)
        
        self.vocab = unique_chars
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def unk_id(self) -> int:
        return self.stoi.get(self.UNK_TOKEN, 0)

    def encode(self, text: str) -> List[int]:
        """Encodes text string to a list of integer token IDs."""
        unk = self.unk_id
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list or tensor of integer token IDs back to a string."""
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        
        chars = []
        for t in tokens:
            if t in self.itos:
                ch = self.itos[t]
                if ch != self.UNK_TOKEN:
                    chars.append(ch)
            else:
                chars.append("")
        return "".join(chars)

    def save(self, filepath: Union[str, Path]):
        """Saves vocabulary mappings to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False, indent=2)

    def load(self, filepath: Union[str, Path]):
        """Loads vocabulary mappings from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}