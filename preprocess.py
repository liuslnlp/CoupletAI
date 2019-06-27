import torch

def create_dataset(seqs, tags, word_to_ix, max_seq_len, pad_ix):
    assert len(seqs) == len(tags)
    num_seqs = len(seqs)
    seqs_tensor = torch.ones(num_seqs, max_seq_len) * pad_ix
    seqs_mask = torch.zeros(num_seqs, max_seq_len)
    tags_tesnor = torch.ones(num_seqs, max_seq_len) * pad_ix
    for i in range(num_seqs):
        seqs_mask[i, : len(seqs[i])] = 1
        for j, word in enumerate(seqs[i]):
            seqs_tensor[i, j] = word_to_ix.get(word, word_to_ix['[UNK]'])
        for j, tag in enumerate(tags[i]):
            tags_tesnor[i, j] = word_to_ix.get(tag, word_to_ix['[UNK]'])
    return seqs_tensor.long(), seqs_mask, tags_tesnor.long()

def create_attention_mask(raw_mask):
    extended_attention_mask = raw_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask.float()
            
