import torch 
from dataclasses import dataclass
import torch.nn as nn 
import math 
import torch.nn.functional as F
# from flash_attn import flash_attn_func



@dataclass
class modelArguments:
    d_model : int = 384
    n_layers : int = 4
    n_heads : int = 8
    vocab_size : int = 32768                       
    device : str = "cuda"
    max_seq_len = 256
    max_batch_size = 4
    dropout = 0.1
    math_layers = 6
    math_dim1 = 256
    math_dim2 = 2048
    se_docode = 768
    enforce_features = 64
    mffLayers = 4
    mffdim = 256

class shared_emb(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model

        # Create a single shared embedding
        self.shared_weight = nn.Parameter(torch.randn(vocab_size, d_model))

        # Initialize the weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.shared_weight)

    def embedding(self, x):
        # Scale embeddings by sqrt(d_model)
        return nn.functional.embedding(x, self.shared_weight) * math.sqrt(self.d_model)

    def linear(self, x):
        # Use the transpose of the shared weight for the pre-softmax linear transformation
        return nn.functional.linear(x, self.shared_weight)

def get_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
    pos_encoding = torch.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding




class model(nn.Module):
    def __init__(self, args : modelArguments) -> None:
        super().__init__()

        assert args.vocab_size != -1, "vocab size should be set."

        self.args = args
        self.shared_emb = shared_emb(args.vocab_size, args.d_model)
        # self.enc_emb = nn.Embedding(args.vocab_size, args.d_model)
        

        self.encoderlayer = nn.TransformerEncoderLayer(d_model=args.d_model, dim_feedforward=args.d_model*4, nhead=args.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=args.n_layers)

        self.decoderlayer = nn.TransformerDecoderLayer(d_model=args.d_model, dim_feedforward=args.d_model*4, nhead=args.n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoderlayer, num_layers=args.n_layers)


        self.norm = nn.LayerNorm(args.d_model)
        self.norm1 = nn.LayerNorm(args.d_model)
        # self.linear = nn.Linear(args.d_model, args.vocab_size, bias = False)
        self.enc = get_positional_encoding(args.max_seq_len, args.d_model).to(args.device)
        # self.softmax = nn.Softmax(-1)



    def forward(self, in_token : torch.tensor, out_token: torch.tensor):

        # (batch, seq_len)
        batch, seq_len = in_token.shape
        inp_embd = self.shared_emb.embedding(in_token)
        # inp_embd = F.dropout(inp_embd, self.args.dropout)
        # print(self.enc.shape)
        inp_embd = inp_embd + self.enc
        # print(freq.shape)
        # prev_layer = inp_embd

        inp_embd1 = self.encoder(inp_embd)
        # for layer in self.enc_layers:
        #     inp_embd = layer(inp_embd)
        #     inp_embd = F.dropout(inp_embd, self.args.dropout)
        #     inp_embd = inp_embd + prev_layer
        #     prev_layer = inp_embd

        inp_embd = self.norm(inp_embd) 

        out_embd = self.shared_emb.embedding(out_token)
        
        out_embd = out_embd + self.enc

        out_len = out_embd.size(1)
        causal_mask = torch.triu(torch.ones(out_len, out_len), diagonal=1).bool()


        out_embd = self.decoder(out_embd, inp_embd, tgt_mask=causal_mask)
        # for layer in self.dec_layers:
        #     out_embd = layer(out_embd, inp_embd)
        #     out_embd = out_embd + prev_layer
        #     out_embd = F.dropout(out_embd, self.args.dropout)
        #     prev_layer = out_embd

        embd = self.norm(out_embd)
        output = self.shared_emb.linear(embd)

        return output                   