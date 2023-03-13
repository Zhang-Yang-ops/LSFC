import torch
import torch.nn as nn
from model1 import BIN_Interaction_Flat


class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats * 4),
            nn.Linear(n_feats * 4, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)
        return x


class LSFC(torch.nn.Module):
    def __init__(self, hidden_size, **config):
        super(LSFC, self).__init__()
        self.model1 = BIN_Interaction_Flat(hidden_size, **config)

        self.lin = nn.Sequential(
            nn.Linear(1705, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )

        self.rmodule = nn.Embedding(86, hidden_size)
        self.lin_block = LinearBlock(hidden_size)

    def forward(self, triples):
        head_pairs, tail_pairs, rels, d_v, p_v, input_mask_d, input_mask_p = triples

        h_final1, t_final1 = self.model1(rels, d_v.long(), p_v.long(), input_mask_d.long(), input_mask_p.long())

        h_final = self.lin(head_pairs.float())
        t_final = self.lin(tail_pairs.float())

        hidden = torch.cat([h_final, h_final1, t_final, t_final1], dim=-1)
        rfeat = self.rmodule(rels)
        logit = (self.lin_block(hidden) * rfeat).sum(-1)
        return logit