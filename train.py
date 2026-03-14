import torch
from model.transformer import BigramLanguageModel

# ---------------- Hyperparameters ----------------
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------------------------------------

torch.manual_seed(1337)

# Load dataset
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loader
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Estimate loss
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize model
model = BigramLanguageModel(vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                            block_size=block_size, dropout=dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save final model checkpoint
torch.save(model.state_dict(), 'checkpoints/final_model.pt')

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=1000)
generated_text = decode(generated_ids[0].tolist())

print("\n--- Generated Text ---\n")
print(generated_text)

# Save generated text
with open('data/generated.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)