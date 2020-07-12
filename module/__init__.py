from .tokenizer import Tokenizer
from .model import *
import argparse

def init_model_by_key(args, tokenizer: Tokenizer):
    key = args.model.lower()
    if key == 'transformer':
        model = Transformer(tokenizer.vocab_size, args.max_seq_len, args.embed_dim, args.hidden_dim, args.n_layer, args.n_head, args.ff_dim, args.embed_drop, args.hidden_drop)
    elif key == 'bilstm':
        model = BiLSTM(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop)
    elif key == 'cnn':
        model = CNN(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.embed_drop)
    elif key == 'bilstmattn':
        model = BiLSTMAttn(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop, args.n_head)
    elif key == 'bilstmcnn':
        model = BiLSTMCNN(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop)
    elif key == 'bilstmconvattres':
        model = BiLSTMConvAttRes(tokenizer.vocab_size, args.max_seq_len, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop, args.n_head)
    else:
        raise KeyError(f"Model `{args.model}` does not exist")
    return model