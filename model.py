from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from cfg import model_cfg
from dataset import alphabet_map


class CavemanGPTAttentionHead(nn.Module):
    def __init__(self, emb_dim, cfg):
        super().__init__()

        hidden_dim = emb_dim * 4

        self.W_q = nn.Parameter(torch.rand(hidden_dim, emb_dim))
        self.W_k = nn.Parameter(torch.rand(hidden_dim, emb_dim))
        self.W_v = nn.Parameter(torch.rand(cfg['output_dim'], emb_dim))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(cfg['dropout'])
        self.d_k = hidden_dim

    def forward(self, token_emb, mask):
        Q = token_emb @ self.W_q.T
        K = token_emb @ self.W_k.T
        V = token_emb @ self.W_v.T

        Q, K, V = Q.T, K.T, V.T

        inner_prod = (K.T @ Q.permute([2, 0, 1]))
        inner_prod /= self.d_k ** 0.5
        inner_prod.masked_fill(mask == 0, float('-inf'))
        softmax = self.softmax(inner_prod)
        return self.dropout(softmax @ V.T)


class CavemanGPTAttentionBlock(nn.Module):
    def __init__(self, emb_dim, cfg):
        super().__init__()

        self.attention_heads = nn.ModuleList([CavemanGPTAttentionHead(emb_dim, cfg) for _ in range(cfg['num_heads'])])

    def forward(self, token_emb, mask):
        mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1]))
        head_outputs = [head(token_emb, mask) for head in self.attention_heads]
        return torch.concat(head_outputs, dim=-1)


class CaveManGPTBlock(nn.Module):
    def __init__(self, emb_dim, cfg):
        super().__init__()

        self.layernorm = nn.LayerNorm(emb_dim)
        self.act_fn = nn.ReLU()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), self.act_fn)
        self.attention = CavemanGPTAttentionBlock(emb_dim, cfg)
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, token_emb, mask):
        h = self.attention(token_emb, mask) + token_emb
        h = self.layernorm(h)
        return self.layernorm(self.dropout(self.mlp(h)) + h)


class CavemanGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_emb = nn.Embedding(cfg['num_embs'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['block_size'], cfg['emb_dim'])

        self.blocks = nn.ModuleList(([CaveManGPTBlock(cfg['emb_dim'], cfg['block_cfg'])
                                                 for i in range(cfg['num_blocks'])]))
        self.lm_head = nn.Linear(cfg['emb_dim'], cfg['num_embs'])
        self.dropout = nn.Dropout(cfg['input_emb_dropout'])

    def forward(self, input_ids, mask, labels=None):
        pos_ids = torch.arange(0, input_ids.shape[-1])
        h = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        h = self.dropout(h)

        for block in self.blocks:
            h = block(token_emb=h, mask=mask)

        if labels is None:
            return (h,)
        else:
            logits = self.lm_head(h)
            loss = F.cross_entropy(F.softmax(logits.flatten(0, 1), -1), labels.flatten(0, 1))
            return (h, loss)

if __name__ == '__main__':

    input_texts = ["abc" * (i+1) for i in range(5)]

    tokenized_texts = [[alphabet_map[x] for x in input_text] for input_text in input_texts]

    max_text_len = max([len(x) for x in tokenized_texts])

    for idx in range(len(tokenized_texts)):
        padding = [0] * (max_text_len - len(tokenized_texts[idx]))
        tokenized_texts[idx] += padding

    input_idx = torch.tensor(tokenized_texts, dtype=torch.long)
    mask = input_idx > 0

    model = CavemanGPT(model_cfg)
    model.forward(input_idx, mask)

