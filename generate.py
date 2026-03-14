"""It does not retrain – only loads the checkpoint."""
import torch
from model.transformer import BigramLanguageModel

# ---------------- Config ----------------
checkpoint_path = 'checkpoints/final_model.pt'
max_new_tokens = 1000  # number of characters to generate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ----------------------------------------

# Load dataset to get vocab
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Initialize model
n_embd = 384
n_head = 6
n_layer = 6
block_size = 256
dropout = 0.2

model = BigramLanguageModel(vocab_size, n_embd=n_embd, n_layer=n_layer,
                            n_head=n_head, block_size=block_size, dropout=dropout).to(device)

# Load checkpoint
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Start generation with a blank context
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=max_new_tokens)
generated_text = decode(generated_ids[0].tolist())

print("\n--- Generated Text ---\n")
print(generated_text)

# Save generated text
with open('data/generated1.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)