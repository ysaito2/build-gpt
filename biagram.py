import torch
import torch.nn as nn
from torch.nn import functional as F

#--------------------------------
# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200

print('='*40)
print('Hyperparameters:')
print(f'  batch_size: {batch_size}')
print(f'  block_size: {block_size}')
print(f'  max_iters: {max_iters}')
print(f'  eval_interval: {eval_interval}')
print(f'  learning_rate: {learning_rate}')
print(f'  eval_iters: {eval_iters}')
print(f"  Using device: {device}")
print('='*40)
#--------------------------------
# data
with open('input.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and val splits
data = torch.tensor(encode(text), dtype=torch.long)
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
class BigramLanguageModel(nn.Module):
    """A simple bigram language model that predicts the next character based on the current character."""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
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
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

#--------------------------------
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


#--------------------------------
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
        
#--------------------------------
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

#--------------------------------
# test
model = BigramLanguageModel(vocab_size=len(chars))
m = model.to(device)
train_model(m)

print('='*40)
print('Generating text...')
print('='*40)
print(generate_text(m))