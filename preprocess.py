import argparse
from pathlib import Path
from typing import Tuple, List, Mapping

import torch
from tqdm import trange

import config
from data_load import load_vocab, load_dataset


def create_dataset(seqs: List[List[str]],
                   tags: List[List[str]],
                   word_to_ix: Mapping[str, int],
                   max_seq_len: int,
                   pad_ix: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert List[str] -> torch.Tensor.
    Returns:
        seqs_tensor: shape=[num_seqs, max_seq_len].
        seqs_mask: shape=[num_seqs, max_seq_len].
        tags_tesnor: shape=[num_seqs, max_seq_len].
    """
    assert len(seqs) == len(tags)
    num_seqs = len(seqs)
    seqs_tensor = torch.ones(num_seqs, max_seq_len) * pad_ix
    seqs_mask = torch.zeros(num_seqs, max_seq_len)
    tags_tesnor = torch.ones(num_seqs, max_seq_len) * pad_ix
    for i in trange(num_seqs):
        seqs_mask[i, : len(seqs[i])] = 1
        for j, word in enumerate(seqs[i]):
            seqs_tensor[i, j] = word_to_ix.get(word, word_to_ix['[UNK]'])
        for j, tag in enumerate(tags[i]):
            tags_tesnor[i, j] = word_to_ix.get(tag, word_to_ix['[UNK]'])
    return seqs_tensor.long(), seqs_mask, tags_tesnor.long()


def save_dataset(seqs_tensor, seqs_mask, tags_tesnor, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(seqs_tensor, path / 'seqs_tensor.pkl')
    torch.save(seqs_mask, path / 'seqs_mask.pkl')
    torch.save(tags_tesnor, path / 'tags_tesnor.pkl')


def create_attention_mask(raw_mask: torch.Tensor) -> torch.Tensor:
    """Convert mask to attention mask.
    """
    extended_attention_mask = raw_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask.float()


def create_transformer_attention_mask(raw_mask: torch.Tensor) -> torch.Tensor:
    """Convert mask to transformer attention mask.
    """
    return (1 - raw_mask).bool()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='tensor_dataset', type=str)
    parser.add_argument("--max_len", default=32, type=int)

    args = parser.parse_args()
    seq_path = f'{config.data_dir}/train/in.txt'
    tag_path = f'{config.data_dir}/train/out.txt'
    vocab_path = f'{config.data_dir}/vocabs'
    max_seq_len = args.max_len

    word_to_ix = load_vocab(vocab_path)
    vocab_size = len(word_to_ix)
    seqs, tags = load_dataset(seq_path, tag_path)
    seqs, masks, tags = create_dataset(seqs, tags, word_to_ix, max_seq_len, word_to_ix['[PAD]'])
    save_dataset(seqs, masks, tags, args.dir)
