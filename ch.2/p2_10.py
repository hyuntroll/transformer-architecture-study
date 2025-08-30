from torch import nn


class PreLayerNormFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src):
        x = self.norm(src)
        x - x + self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        return x