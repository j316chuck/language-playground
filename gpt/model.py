import math 
import torch

from torch import nn 
from torch.nn import functional as F

@torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, dropout=0.0, **kwargs):
        super().__init__()
        self.n_vocab = vocab_size
        self.n_embed = embed_dim
        self.n_block = block_size
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

        self.token_embed_block = torch.nn.Embedding(self.n_vocab, self.n_embed)
        self.pos_embed_block = torch.nn.Embedding(self.n_block, self.n_embed)
        self.intermediate_layer = nn.Linear(self.n_embed, self.n_embed)
        self.projection = nn.Linear(self.n_embed, self.n_vocab)
        
        self.n_params = sum(p.numel() for p in self.parameters())
        print("Number of parameters", self.n_params)
    
    def forward(self, inputs, targets=None):
        device = inputs.device
        b, s = inputs.size()  # (batch_size, seq_length)
        pos = torch.arange(0, s, dtype=torch.long, device=device).unsqueeze(0) # shape (1, seq_length)
        
        inp_proj = self.token_embed_block(inputs)  # (batch_size, seq_length, n_embed)
        pos_proj = self.pos_embed_block(pos)  # (1, seq_length, n_embed)
        x = inp_proj + pos_proj  # (batch_size, seq_length, n_embed)
        x = new_gelu(x)
        x = self.dropout_1(x)
        x = self.intermediate_layer(x)
        x = new_gelu(x)
        x = self.dropout_2(x)
        logits = self.projection(x)  # (batch_size, seq_length, vocab_size)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='mean')
        else:
            loss = None
        return logits, loss
        