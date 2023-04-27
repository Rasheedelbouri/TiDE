import torch
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, size_in, size_out, p_dropout=0.5, device='cpu'):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.device = device

        self.nonlin = nn.Linear(size_in, size_out, device=self.device)
        self.lin = nn.Linear(size_out, size_out, device=self.device)

        self.branch = nn.Linear(size_in, size_out, device=self.device)

        self.dropout = nn.Dropout(p=p_dropout)
        self.layernorm = nn.LayerNorm(size_out)

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
                 dropout: float = 0.5,
                 attribs_size: int = None,
                 device: str = 'cpu'):
        
        super().__init__()

        self.lookback = lookback
        self.device  = device

        self.size_in = size_in
        self.size_out = size_out
        self.dropout = dropout

        self.horizon: int = self.size_in - self.lookback

        self.enc_in_size: int = self.lookback + self.size_in
        if attribs_size is not None:
            self.enc_in_size += self.attribs_size


        self.in_res = ResBlock(size_in = self.size_in,
                               size_out=self.size_in,
                               p_dropout=self.dropout,
                               device = self.device)
        
        self.encoder = ResBlock(size_in = self.enc_in_size,
                                size_out = self.size_out,
                                p_dropout=self.dropout,
                                device = self.device)
        
        self.decoder = ResBlock(size_in = self.size_out,
                                size_out = self.horizon,
                                p_dropout=self.dropout,
                                device = self.device)
        
        self.output_res = ResBlock(size_in = self.horizon,
                                   size_out = self.horizon,
                                   p_dropout = self.dropout,
                                   device = self.device)
        
        self.y_transform = nn.Linear(self.lookback, self.horizon, device=self.device)

    def forward(self, X: torch.tensor, y: torch.tensor, attribs: torch.tensor = None):

        transform = self.in_res(X)
        joined = torch.hstack((transform, y[:self.lookback]))
        if self.attribs_size is not None:
            joined = torch.hstack((joined, attribs))
        
        Z = self.encoder(joined)
        outs = self.decoder(Z)

        outs = torch.vstack((outs, transform[-self.horizon:]))

        outputs = self.output_res(outs)

        return self.y_transform(y) + outputs



