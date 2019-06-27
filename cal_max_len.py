import data_load
import config
from typing import List


def cal_max_seq_len(seqs: List[List[str]]) -> int:
    """Calculate the max length of a group of sequence.
    """
    cur_max_len = 0
    for seq in seqs:
        if len(seq) > cur_max_len:
            cur_max_len = len(seq)
    return cur_max_len


def cal_total_max_len(seqss: List[List[List[str]]]) -> int:
    """Calculate the max_seq_len in both train dataset and test dataset.
    """
    cur_max_len = 0
    cur_len = 0
    for seqs in seqss:
        cur_len = cal_max_seq_len(seqs)
        if cur_len > cur_max_len:
            cur_max_len = cur_len
    return cur_max_len


def main():
    fdir = config.data_dir
    train_data, _ = data_load.load_dataset(
        f'{fdir}/train/in.txt', f'{fdir}/train/out.txt')
    test_data, _ = data_load.load_dataset(
        f'{fdir}/test/in.txt', f'{fdir}/test/out.txt')
    print(cal_total_max_len([train_data, test_data]))


if __name__ == "__main__":
    main()
