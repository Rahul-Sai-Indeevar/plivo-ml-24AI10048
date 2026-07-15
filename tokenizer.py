import json
import os
from collections import Counter

class CharTokenizer:
    def __init__(self):
        # 0-255 reserved for raw UTF-8 bytes (lossless fallback)
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 256

    def train(self, text, max_vocab=1500):
        # Find frequent multi-byte characters (like Hindi) and give them single tokens
        chars = Counter(text)
        for ch, _ in chars.most_common():
            if len(ch.encode('utf-8')) > 1:
                idx = self.vocab_size
                self.char_to_id[ch] = idx
                self.id_to_char[idx] = ch
                self.vocab_size += 1
                if self.vocab_size >= max_vocab:
                    break

    def encode(self, text):
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                ids.extend(list(ch.encode("utf-8")))
        return ids

    def decode(self, ids):
        b_out = bytearray()
        for i in ids:
            if i < 256:
                b_out.append(i)
            else:
                b_out.extend(self.id_to_char[i].encode("utf-8"))
        return bytes(b_out).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id}, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.vocab_size = 256 + len(self.char_to_id)
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}

def load(path="tokenizer.json"):
    tok = CharTokenizer()
    if os.path.exists(path):
        tok.load(path)
    return tok