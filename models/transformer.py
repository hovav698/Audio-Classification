import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_padding_mask(seq):
  mask=(seq==0)[:,:,0]
  return torch.transpose(mask,0,1)

#adding positional elemnts to the data. If there is obvious relationship between different parts of the audio files,
#using positinal encoding will help to identify those relation.
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#the transformer model
class Transformer_Model(torch.nn.Module):
  def __init__(self,K,d_model=128,num_head=8):
    super(Transformer_Model,self).__init__()

    self.pos_encoding=PositionalEncoding(d_model).to(device)
    self.lin1=torch.nn.Linear(128,128)
    self.relu1=torch.nn.ReLU()
    self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head)
    self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
    self.avgpool1d=torch.nn.AvgPool1d(3, stride=3)
    self.flatten=torch.nn.Flatten()
    self.lin2=torch.nn.Linear(21882,K)

  def forward(self,x):
    mask=get_padding_mask(x)
    out=self.lin1(x)
    out+=self.pos_encoding(out)
    out=self.relu1(x)
    out=self.transformer_encoder(x,src_key_padding_mask=mask)
    out=self.avgpool1d(out)
    out=self.avgpool1d(out)
    out=self.flatten(out)
    dim=out.shape[-1]
    out=self.lin2(out)

    return out