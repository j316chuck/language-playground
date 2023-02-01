import torch
from torch import nn 
from torch.nn import functional as F

def new_gelu(x):
    """
    Implementation of the GELU activation function
    """
    return 0.5 * x * (1.0 * torch.tanh())

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, dropout=0.0, **kwargs):
        super().__init__()
        self.n_vocab = vocab_size
        self.n_embed = embed_dim
        self.n_block = block_size
        self.dropout = dropout

        self.token_embed_block = torch.nn.Embedding(self.n_vocab, self.n_embed)
        self.pos_embed_block = torch.nn.Embedding(self.n_block, self.n_embed)
        self.intermediate_dropout = nn.Dropout(p=self.dropout)
        self.projection = nn.Linear(self.n_embed, self.n_vocab)
        
        n_params = sum(p.numel() for p in self.parameters())
        print("Number of parameters", n_params)
    
    def forward(self, inputs, targets=None):
        device = inputs.device
        b, s = inputs.size()  # (batch_size, seq_length)
        pos = torch.arange(0, s, dtype=torch.long, device=device).unsqueeze(0) # shape (1, seq_length)
        
        inp_proj = self.token_embed_block(inputs)  # (batch_size, seq_length, n_embed)
        pos_proj = self.pos_embed_block(pos)  # (1, seq_length, n_embed)
        intermediate = inp_proj + pos_proj  # (batch_size, seq_length, n_embed)
        intermediate = self.intermediate_dropout(intermediate)
        logits = self.projection(intermediate)  # (batch_size, seq_length, vocab_size)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='mean')
        else:
            loss = None
        return logits, loss
        