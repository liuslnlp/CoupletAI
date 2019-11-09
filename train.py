import config
from model import CNNBiLSTMAtt, TraForEncoder
from data_load import load_dataset, load_vocab, load_tensor_dataset
from preprocess import create_dataset, create_attention_mask, create_transformer_attention_mask
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import argparse
from pathlib import Path


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_dataset(path, batch_size):
    # extended_attention_mask = create_attention_mask(masks)
    seqs, masks, tags = load_tensor_dataset(path)
    extended_attention_mask = create_transformer_attention_mask(masks)
    dataset = TensorDataset(seqs, extended_attention_mask, tags)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_model(model, output_dir, epoch):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.module if hasattr(model, 'module') else model
    filename = output_dir / f"transformer_{epoch:02}.pkl"
    logger.info(f'***** Save model `{filename}` *****')
    torch.save(model.state_dict(), filename)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=768, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    # parser.add_argument("--max_len", default=32, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", default='O1', type=str)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dir", default='tensor_dataset', type=str)

    return parser.parse_args()

# H 4 N 3 Loss:5.20
# H 2 N 2 Loss:5.20


def main():
    vocab_path = f'{config.data_dir}/vocabs'

    args = get_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    embed_dim = config.embed_dim
    hidden_dim = config.hidden_dim
    output_dir = config.ouput_dir

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    logger.info(f"***** Loading vocab *****")
    word_to_ix = load_vocab(vocab_path)
    vocab_size = len(word_to_ix)

    logger.info(f"***** Initializing dataset *****")
    train_dataloader = init_dataset(args.dir, batch_size)



    logger.info(f"***** Training *****")
    # model = CNNBiLSTMAtt(vocab_size, embed_dim, hidden_dim)
    model = TraForEncoder(vocab_size, embed_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    if args.fp16:
        try:
            from apex import amp
            amp.register_half_function(torch, 'einsum')
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    loss_func = nn.CrossEntropyLoss(ignore_index=word_to_ix['[PAD]'])
    logger.info(f"Num GPU {torch.cuda.device_count()}")
    for epoch in range(epochs):
        logger.info(f"***** Epoch {epoch} *****")
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            seq_ids, exted_att_mask, tag_ids = batch
            logits = model(seq_ids, exted_att_mask)
            loss = loss_func(logits.view(-1, vocab_size), tag_ids.view(-1))
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if step % 100 == 0:
                logger.info(
                    f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item()}")
        save_model(model, output_dir, epoch + 1)


if __name__ == '__main__':
    main()
