import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size=256, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) 
        self.dropout = nn.Dropout(dropout)
# projection is a linear transformation that takes the concatenated outputs of all the heads and maps it back to the original embedding dimension.
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    # self attention is the communication once they've gathered all the data
    # now they need think on that data individually and that's what the feed forward is for
    def __init__(self, n_embd,dropout):
        super().__init__()
        self.net = nn.Sequential(
# we multiply by 4 because we want to increase the dimensionality of the feedforward layer. This is a common practice in transformer architectures, where the feedforward layer typically has a larger dimensionality than the input embedding size. The idea is that by increasing the dimensionality, the model can learn more complex representations and capture more intricate patterns in the data.
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, num_heads, dropout, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size=n_embd//num_heads, n_embd=n_embd,dropout=dropout) #communication
        self.ffwd = FeedForward(n_embd, dropout) #computation
        self.ln1 = nn.LayerNorm(n_embd) # Layer normalization is a technique used to normalize the inputs across the features. It helps to stabilize and accelerate the training of deep neural networks by reducing internal covariate shift. In this code, we apply layer normalization before the self-attention and feedforward layers to ensure that the inputs to these layers have a consistent distribution, which can improve convergence and overall performance of the model.
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        #here we applied residual connections around both the self-attention and the feedforward layers. This is a common technique in deep learning that helps to mitigate the vanishing gradient problem and allows for better flow of gradients during training.
        return x
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size, n_embd=384, n_layer=6, n_head=6, block_size=256, dropout=0.2):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=n_head, dropout=dropout, block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        #x = self.sa_head(x) # apply one head of self-attention (B,T,C)
        #x = self.ffwd(x) # apply feedforward (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None: 
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens so position embeddings don't go out of bounds
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

