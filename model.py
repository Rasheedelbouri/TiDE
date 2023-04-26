import torch
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, size_in, size_out, p_dropout=0.5):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out

        self.nonlin = nn.Linear(size_in, size_out)
        self.lin = nn.Linear(size_out, size_out)

        self.branch = nn.Linear(size_in, size_out)

        self.dropout = nn.Dropout(p=p_dropout)
        self.layernorm = nn.LayerNorm(size_in)

        self.relu = nn.ReLU()

    def forward(self, X):
        z = self.relu(self.nonlin(X))
        z = self.dropout(self.lin(z))
        added = self.branch(X) + z
        return self.layernorm(added)
    
class TiDE(nn.Module):

    def __init__(self,
                 size_in: int,
                 size_out: int,
                 lookback: int,
                 dropout: int = 0.5,
                 attribs_size: int = None):
        
        super().__init__()

        self.lookback = lookback

        self.size_in = size_in
        self.size_out = size_out
        self.dropout = dropout

        self.enc_in_size = self.lookback + self.size_in
        if attribs_size is not None:
            self.enc_in_size += self.attribs_size


        self.in_res = ResBlock(size_in = self.size_in,
                               size_out=self.size_in,
                               p_dropout=self.dropout)
        
        self.encoder = ResBlock(size_in = self.enc_in_size,
                                size_out = self.size_out,
                                p_dropout=self.dropout)
        
        self.decoder = ResBlock(size_in = self.size_out,
                                size_out = self.size_in - self.lookback,
                                p_dropout=self.dropout)
        
        self.output_res = ResBlock(size_in = self.size_in - self.lookback,
                                   size_out = self.size_in - self.lookback,
                                   p_dropout = self.dropout)
        
        self.y_transform = nn.Linear(self.lookback, self.size_in - self.lookback)

    def forward(self, X: torch.tensor, y: torch.tensor, attribs: torch.tensor = None):

        transform = self.in_res(X)
        joined = torch.hstack((transform, y[:self.lookback]))
        if self.attribs_size is not None:
            joined = torch.hstack((joined, attribs))
        
        Z = self.encoder(joined)
        outs = self.decoder(Z)

        outs = torch.vstack((outs, transform[-self.lookback:]))

        outputs = self.output_res(outs)

        return self.y_transform(y) + outputs



