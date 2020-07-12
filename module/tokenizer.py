from typing import List
from pathlib import Path
import torch


class Tokenizer(object):
    def __init__(self):
        self.token_to_ix = {}
        self.ix_to_token = {}

    @property
    def vocab_size(self):
        return len(self.token_to_ix)

    @property
    def pad_id(self):
        return self.token_to_ix['[PAD]']

    @property
    def unk_id(self):
        return self.token_to_ix['[UNK]']

    def build(self, vocab_file: str or Path):
        """Build tokenizer from a vocab file.
        """
        if isinstance(vocab_file, str):
            vocab_file = Path(vocab_file)
        token_to_ix = {'[PAD]': 0, '[UNK]': 1}
        with vocab_file.open('r', encoding='utf-8') as f:
            for token in f.readlines():
                token = token.rstrip("\n")
                if token not in token_to_ix:
                    token_to_ix[token] = len(token_to_ix)
        self.token_to_ix = token_to_ix
        self.ix_to_token = {v: k for k, v in token_to_ix.items()}

    def save_pretrained(self, filename: str or Path):
        info_dict = {
            'token_to_ix': self.token_to_ix,
            'ix_to_token': self.ix_to_token,
        }
        if isinstance(filename, str):
            filename = Path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)
        torch.save(info_dict, filename)

    @classmethod
    def from_pretrained(cls, filename: str or Path):
        info_dict = torch.load(filename)
        token_to_ix = info_dict['token_to_ix']
        ix_to_token = info_dict['ix_to_token']
        kls = cls()
        kls.token_to_ix = token_to_ix
        kls.ix_to_token = ix_to_token
        return kls

    def encode(self, sent: str):
        tokens = list(sent)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: List[int]):
        tokens = self.convert_ids_to_tokens(ids, True)
        return "".join(tokens)

    def convert_token_to_id(self, token: str):
        return self.token_to_ix.get(token, self.token_to_ix['[UNK]'])

    def convert_id_to_token(self, id: int):
        return self.ix_to_token[id]

    def convert_tokens_to_ids(self, tokens: List[str]):
        return [self.convert_token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int], ignore_pad: bool = False):
        tokens = []
        for t in ids:
            if ignore_pad and t != self.pad_id:
                tokens.append(self.ix_to_token[t])
        return tokens
