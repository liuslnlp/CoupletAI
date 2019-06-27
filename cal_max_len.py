import data_load

def cal_max_seq_len(seqs):
    cur_max_len = 0
    for seq in seqs:
        if len(seq) > cur_max_len:
            cur_max_len = len(seq)
    return cur_max_len


def cal_total_max_len(seqss):
    cur_max_len = 0
    cur_len = 0
    for seqs in seqss:
        cur_len = cal_max_seq_len(seqs)
        if cur_len > cur_max_len:
            cur_max_len = cur_len
    return cur_max_len

def main():
    train_data, _ = data_load.load_dataset('couplet_data/train/in.txt', 'couplet_data/train/out.txt')
    test_data, _ = data_load.load_dataset('couplet_data/test/in.txt', 'couplet_data/test/out.txt')
    print(cal_total_max_len([train_data, test_data]))

if __name__ == "__main__":
    main()