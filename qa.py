import os
from typing import List, Mapping

import torch
import torch.nn as nn

import config
from data_load import load_vocab
from model import TraForEncoder
from preprocess import create_transformer_attention_mask


class QAContext(object):
    def __init__(self, model: nn.Module, word_to_ix: Mapping[str, int], device: torch.device):
        self.model = model
        self.word_dict = word_to_ix
        self.device = device
        self.ix2word = {v: k for k, v in self.word_dict.items()}
        self.model.to(self.device)
        self.model.eval()
        self.model = self._build_traced_script_module()

    def _build_traced_script_module(self):
        example = torch.ones(1, 3).long().to(self.device)
        mask = create_transformer_attention_mask(torch.ones_like(example).to(self.device))
        return torch.jit.trace(self.model, (example, mask))

    def predict(self, seq: List[List[str]]) -> str:
        seq = [self.word_dict.get(word, self.word_dict['[UNK]'])
               for word in seq]
        seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = create_transformer_attention_mask(torch.ones_like(seq).to(self.device))

        logits = self.model(seq, attention_mask)
        out_ids = torch.argmax(logits.squeeze(0), dim=-1)
        out_seq = [self.ix2word[idx.item()] for idx in out_ids]
        return ''.join(out_seq)

    def run_console_qa(self, end_flag: str):
        while True:
            question = input("上联：")
            if question == end_flag.lower():
                print("Thank you!")
                break
            answer = self.predict(question)
            print(f"下联：{answer}")


def create_qa_context(model_path: str, word_to_ix_path: str,
                      embed_dim: int, hidden_dim: int, device) -> QAContext:
    word_dict = load_vocab(word_to_ix_path)
    vocab_size = len(word_dict)
    model = TraForEncoder(vocab_size, embed_dim, hidden_dim)
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(model_path))
    return QAContext(model, word_dict, device)


if __name__ == "__main__":
    device = torch.device('cpu')
    output_dir = config.ouput_dir
    vocab_path = f'./{config.data_dir}/vocabs'
    model_path = max(os.listdir(output_dir))
    print(f'Model filename: {model_path}')
    ctx = create_qa_context(f'./{output_dir}/{model_path}', vocab_path, config.embed_dim, config.hidden_dim, device)
    ctx.run_console_qa(end_flag='q')
