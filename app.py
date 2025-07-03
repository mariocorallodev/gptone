# app.py
# Interfaccia Gradio – carica model.pt addestrato su mindfulness IT

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import math

# Configurazione
block_size = 64
n_embed = 384
n_heads = 8
n_layers = 6
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Caricamento corpus per dizionario
with open("mindfulness_corpus_it.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

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

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# Caricamento modello già addestrato
model = MiniGPT().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# Interfaccia Gradio
def genera_testo(prompt, max_tokens):
    if not prompt:
        prompt = "La"
    context = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long).to(device)
    out = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    return decode(out)

gr.Interface(
    fn=genera_testo,
    inputs=[
        gr.Textbox(label="Prompt iniziale", placeholder="Es. La consapevolezza"),
        gr.Slider(50, 500, value=200, step=10, label="Lunghezza testo generato")
    ],
    outputs=gr.Textbox(label="Testo generato"),
    title="Mini GPT Mindfulness IT",
    description="GPT addestrato su 30M caratteri italiani. Inserisci un prompt per generare testo mindful."
).launch()
