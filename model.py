import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class BiLSTMLayer(nn.Module):
    """Bi-LSTM layer.

    This module implements a Bi-LSTM layer.  
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x.shape=(batch_size, max_seq_len, input_dim)
        return self.bilstm(x)

class CNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x.shape=(batch_size, max_seq_len, input_dim)
        # cnn_in.shape=(batch_size, input_dim, max_seq_len)
        cnn_in = x.permute(0, 2, 1)
        # self.conv(cnn_in).shape=(batch_size, hidden_dim, max_seq_len)
        # cnn_out.shape=(batch_size, max_seq_len, hidden_dim)
        cnn_out = self.conv(cnn_in).permute(0, 2, 1)
        return F.relu(cnn_out)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: Optional[float] = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: Optional[int] = 4, dropout_prob: Optional[float] = 0.2):
        super().__init__()
        self.num_att_heads = num_heads
        self.att_head_size = hidden_dim // num_heads
        self.all_head_size = hidden_dim

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # x.shape=(batch_size, max_seq_len, all_head_size<hidden_dim>)
        # new_x_shape=(batch_size, max_seq_len, num_att_heads, att_head_size)
        new_x_shape = x.size()[:-1] + (self.num_att_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        # return shape=(batch_size, num_att_heads, max_seq_len, att_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # hidden_states.shape=(batch_size, max_seq_len, hidden_dim)
        # attention_mask.shape=(batch_size, 1, 1, max_seq_len)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # query_layer.shape=(batch_size, num_att_heads, max_seq_len, att_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # key_layer.transpose(-1, -2).shape=(batch_size, num_att_heads, att_head_size, max_seq_len) 
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.att_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores.shape==(batch_size, num_att_heads, max_seq_len, max_seq_len)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs.shape==(batch_size, num_att_heads, max_seq_len, max_seq_len)
        attention_probs = self.dropout(attention_probs)

        # context_layer.shape=(batch_size, num_att_heads, max_seq_len, att_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer.shape=(batch_size, max_seq_len, num_att_heads, att_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape=(batch_size, max_seq_len, all_head_size)
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer




class CNNBiLSTMAtt(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.bilstm = BiLSTMLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.att = SelfAttentionLayer(hidden_dim)
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x, exted_att_mask):
        embeds = self.embedding(x)
        cnn_out = self.conv(embeds)
        lstm_out, _ = self.bilstm(cnn_out)
        normout = self.norm(lstm_out + self.att(lstm_out, exted_att_mask))
        return self.hidden2tag(normout)
