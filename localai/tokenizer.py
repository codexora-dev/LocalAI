import json
from pathlib import Path
from localai.experiment import write_json


class CharTokenizer:
    PAD = "<pad>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.special_tokens = [self.PAD, self.EOS, self.UNK]

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def eos_id(self):
        return self.stoi[self.EOS]

    def build_vocab(self, texts):
        characters = set()

        for text in texts:
            characters.update(text)

        vocabulary = (
            self.special_tokens
            + sorted(characters)
        )

        self.stoi = {
            token: index
            for index, token in enumerate(vocabulary)
        }

        self.itos = {
            index: token
            for token, index in self.stoi.items()
        }

    def encode(self, text, add_eos=False):
        if not self.stoi:
            raise RuntimeError("어휘 사전이 없습니다.")
        
        unk_id = self.stoi[self.UNK]

        ids = [
            self.stoi.get(char, unk_id)
            for char in text
        ]

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, skip_special=True):
        result = []

        for token_id in ids:
            token = self.itos.get(
                int(token_id), self.UNK
            )

            if token == self.EOS:
                if skip_special:
                    break

            if skip_special and token in self.special_tokens:
                continue

            result.append(token)

        return "".join(result)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True, exist_ok=True
        )

        write_json(path, self.stoi)

    @classmethod
    def load(cls, path):
        tokenizer = cls()

        with open(path, "r", encoding="utf-8") as f:
            tokenizer.stoi = json.load(f)

        tokenizer.itos = {
            index: token
            for token, index in tokenizer.stoi.items()
        }

        return tokenizer
