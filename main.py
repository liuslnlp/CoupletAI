import argparse
import logging
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from nltk.translate.bleu_score import sentence_bleu

from module.model import BiLSTM, Transformer, CNN, BiLSTMAttn, BiLSTMCNN, BiLSTMConvAttRes
from module import Tokenizer, init_model_by_key
from module.metric import calc_bleu, calc_rouge_l

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=768, type=int)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("-m", "--model", default='transformer', type=str)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", default='O1', type=str)
    parser.add_argument("--max_grad_norm", default=3.0, type=float)
    parser.add_argument("--dir", default='dataset', type=str)
    parser.add_argument("--output", default='output', type=str)
    parser.add_argument("--logdir", default='runs', type=str)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--ff_dim", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)

    parser.add_argument("--test_epoch", default=1, type=int)
    parser.add_argument("--save_epoch", default=10, type=int)

    parser.add_argument("--embed_drop", default=0.2, type=float)
    parser.add_argument("--hidden_drop", default=0.1, type=float)
    return parser.parse_args()

def auto_evaluate(model, testloader, tokenizer):
    bleus = []
    rls = []
    device = next(model.parameters()).device
    model.eval()
    for step, batch in enumerate(testloader):
        input_ids, masks, lens = tuple(t.to(device) for t in batch[:-1])
        target_ids = batch[-1]
        with torch.no_grad():
            logits = model(input_ids, masks)
            # preds.shape=(batch_size, max_seq_len)
            _, preds = torch.max(logits, dim=-1) 
        for seq, tag in zip(preds.tolist(), target_ids.tolist()):
            seq = list(filter(lambda x: x != tokenizer.pad_id, seq))
            tag = list(filter(lambda x: x != tokenizer.pad_id, tag))
            bleu = calc_bleu(seq, tag)
            rl = calc_rouge_l(seq, tag)
            bleus.append(bleu)
            rls.append(rl)
    return sum(bleus) / len(bleus), sum(rls) / len(rls)

def predict_demos(model, tokenizer:Tokenizer):
    demos = [
        "马齿草焉无马齿", "天古天今，地中地外，古今中外存天地", 
        "笑取琴书温旧梦", "日里千人拱手划船，齐歌狂吼川江号子",
        "我有诗情堪纵酒", "我以真诚溶冷血",
        "三世业岐黄，妙手回春人共赞"
    ]
    sents = [torch.tensor(tokenizer.encode(sent)).unsqueeze(0) for sent in demos]
    model.eval()
    device = next(model.parameters()).device
    for i, sent in enumerate(sents):
        sent = sent.to(device)
        with torch.no_grad():
            logits = model(sent).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = tokenizer.decode(pred)
        logger.info(f"上联：{demos[i]}。 预测的下联：{pred}")

def save_model(filename, model, args, tokenizer):
    info_dict = {
        'model': model.state_dict(),
        'args': args,
        'tokenzier': tokenizer
    }
    torch.save(info_dict, filename)

def run():
    args = get_args()
    fdir = Path(args.dir)
    tb = SummaryWriter(args.logdir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(args)
    logger.info(f"loading vocab...")
    tokenizer = Tokenizer.from_pretrained(fdir / 'vocab.pkl')
    logger.info(f"loading dataset...")
    train_dataset = torch.load(fdir / 'train.pkl')
    test_dataset = torch.load(fdir / 'test.pkl')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    logger.info(f"initializing model...")
    model = init_model_by_key(args, tokenizer)
    model.to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.fp16:
        try:
            from apex import amp
            amp.register_half_function(torch, 'einsum')
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    logger.info(f"num gpu: {torch.cuda.device_count()}")
    global_step = 0
    for epoch in range(args.epochs):
        logger.info(f"***** Epoch {epoch} *****")
        model.train()
        t1 = time.time()
        accu_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, lens, target_ids = batch
            logits = model(input_ids, masks)
            loss = loss_function(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            accu_loss += loss.item()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if step % 100 == 0:
                tb.add_scalar('loss', loss.item(), global_step)
                logger.info(
                    f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item()}")
            global_step += 1
        scheduler.step(accu_loss)
        t2 = time.time()
        logger.info(f"epoch time: {t2-t1:.5}, accumulation loss: {accu_loss:.6}")
        if (epoch + 1) % args.test_epoch == 0:
            predict_demos(model, tokenizer)
            bleu, rl = auto_evaluate(model, test_loader, tokenizer)
            logger.info(f"BLEU: {round(bleu, 9)}, Rouge-L: {round(rl, 8)}")
        if (epoch + 1) % args.save_epoch == 0:
            filename = f"{model.__class__.__name__}_{epoch + 1}.bin"
            filename = output_dir / filename
            save_model(filename, model, args, tokenizer)

if __name__ == "__main__":
    run()