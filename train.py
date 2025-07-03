# train.py
# Addestra MiniGPT su corpus italiano mindfulness (30M caratteri) e salva checkpoint

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Configurazione modello
block_size = 64
batch_size = 16
n_embed = 384
n_heads = 8
n_layers = 6
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Caricamento corpus
with open("mindfulness_corpus_it.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

def get_batch(split):
    src = train_data if split == 'train' else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# Definizione modello
class SelfAttention(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.key = nn.Linear(n_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=device)) == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa = SelfAttention(n_embed, n_heads)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Training
model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
total_steps = 20000
checkpoint_interval = 2000

print(f"⏳ Inizio training ({total_steps} step)...")
for step in range(1, total_steps + 1):
    x, y = get_batch('train')
    logits = model(x)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"[{step}/{total_steps}] loss: {loss.item():.4f}")

    if step % checkpoint_interval == 0:
        ckpt_path = f"model_{step}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Salvato: {ckpt_path}")

torch.save(model.state_dict(), "model.pt")
print("✅ Fine training. Salvato model.pt")
