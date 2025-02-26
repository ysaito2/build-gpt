import torch
import torch.nn as nn
from torch.nn import functional as F

#--------------------------------
# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4  # learning rate set lower since self attention cannot tolerate high learning rates
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_emb = 384
n_head = 6
n_layers = 6
vocab_size = 65
dropout = 0.2

print('='*40)
print(' Hyperparameters:')
print('='*40)
print(f'  batch_size: {batch_size}')
print(f'  block_size: {block_size}')
print(f'  max_iters: {max_iters}')
print(f'  eval_interval: {eval_interval}')
print(f'  learning_rate: {learning_rate}')
print(f'  eval_iters: {eval_iters}')
print(f"  Using device: {device}")
print(f'  n_emb: {n_emb}')
print(f'  vocab_size: {vocab_size}')
print('='*40)
#--------------------------------
# data
with open('input.txt', 'r') as f:
    text = f.read()

MASK_TOKENS = ["<G1>", "<G2>"]
chars = sorted(list(set(text))) + MASK_TOKENS
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and val splits
data = torch.tensor(encode(text), 
                    dtype=torch.long)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

#--------------------------------
# batching
def get_batch(split):
    """Generate a small batch of data of inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#--------------------------------
# model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

#--------------------------------
# multi-head attention
# basically just multiple heads running in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # project the residual pathway back to the original pathway
        return out
        
#--------------------------------
# feed-forward network
class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb), # projection layer back to the residual pathway
            nn.Dropout(dropout), # add dropout to prevent overfitting
        )

    def forward(self, x):
        return self.net(x)

#--------------------------------
# block
# a single transformer block

# note: the results are not great when you naively just stack 3 blocks together
# solution: add residual connections

# why does it work:
# the residual connection adds a seperate pathway for the gradients to flow through 
# initially, the seperate pathway has no gradients, so it does not contribute to the loss
# but gradually the pathway will contribute to the loss
# its effective in training deep models, particularly ones that sufffer from vanishing gradients

# why it helps:
# instead of using the entire network to learn transormation from x to H(x), 
# residual blocks learn only the residual (difference) R(x) between input x and output H(x).


class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # apply the linear norm before both the self-attention and feed-forward network
        x = x + self.ffwd(self.ln2(x)) 
        return x


# --------------------------------
# bigram language model
class BigramLanguageModel(nn.Module):
    """A simple bigram language model that predicts the next character based on the current character."""
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        #self.ffwd = FeedForward(n_emb)
        #self.sa_head = MultiHeadAttention(4, n_emb//4) # i.e. 4 heads of 8 dimensional self-attention
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self.ln_f = nn.LayerNorm(n_emb)
        self.blocks = nn.Sequential(
            *[Block(n_emb, n_head) for _ in range(n_layers)],
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocab_size)
        
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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --------------------------------
# training loop
def train_model(model):
    """Train the bigram language model."""
    print('='*40)
    print('Training started...')
    print('='*40)

    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for step in range(max_iters):
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {step} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")


# --------------------------------
# evaluation 
def estimate_loss(model):
    """Estimate the loss of the model."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out
        
# --------------------------------
# generation
def generate_text(model, max_new_tokens=300):
    """Generate text using the trained model."""
    # Create initial context tensor exactly like the notebook example
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    idx = idx.to(device)
    # Generate tokens using the model
    generated_indices = model.generate(idx=idx, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(generated_indices)

# --------------------------------
# test
model = BigramLanguageModel()
m = model.to(device)
train_model(m)

#--------------------------------
# display the generated text
generated_text = generate_text(m)
print('='*40)
print('Generating text...')
print('='*40)
print(generated_text)

#--------------------------------
# save the generated text in a text file
with open('generated_text.txt', 'w') as f:
    f.write(generated_text)
print('Generated text saved to generated_text.txt')

