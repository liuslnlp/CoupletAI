import config
from model import CNNBiLSTMAtt
from data_load import load_dataset, load_vocab
from preprocess import create_dataset, create_attention_mask
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import argparse


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_dataset(seq_path, tag_path, word_to_ix, max_seq_len, batch_size):
    seqs, tags = load_dataset(seq_path, tag_path)
    seqs, masks, tags = create_dataset(
        seqs, tags, word_to_ix, max_seq_len, word_to_ix['[PAD]'])
    extended_attention_mask = create_attention_mask(masks)
    dataset = TensorDataset(seqs, extended_attention_mask, tags)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_model(model, output_dir, epoch):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"cnn_lstm_att_{epoch:02}.pkl")
    logger.info(f'***** Save model `{filename}` *****')
    torch.save(model.state_dict(), filename)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--max_len", default=32, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    return parser.parse_args()


def main():
    seq_path = f'{config.data_dir}/train/in.txt'
    tag_path = f'{config.data_dir}/train/out.txt'
    vocab_path = f'{config.data_dir}/vocabs'

    args = get_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    max_seq_len = args.max_len
    
    embed_dim = config.embed_dim
    hidden_dim = config.hidden_dim
    output_dir = config.ouput_dir

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    logger.info(f"***** Loading vocab *****")
    word_to_ix = load_vocab(vocab_path)
    vocab_size = len(word_to_ix)

    logger.info(f"***** Initializing dataset *****")
    train_dataloader = init_dataset(
        seq_path, tag_path, word_to_ix, max_seq_len, batch_size)

    logger.info(f"***** Training *****")
    model = CNNBiLSTMAtt(vocab_size, embed_dim, hidden_dim)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(ignore_index=word_to_ix['[PAD]'])

    for epoch in range(epochs):
        logger.info(f"***** Epoch {epoch} *****")
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            seq_ids, exted_att_mask, tag_ids = batch
            logits = model(seq_ids, exted_att_mask)
            loss = loss_func(logits.view(-1, vocab_size), tag_ids.view(-1))
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                logger.info(
                    f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item()}")
        save_model(model, output_dir, epoch + 1)


if __name__ == '__main__':
    main()
