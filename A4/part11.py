#!/usr/bin/env python
# part11.py  – Zero-shot evaluation on CLEVR type-B (or type-A) using the Part-8 model
# -------------------------------------------------------------------------------
# NOTE:  The entire model definition below is **identical** to Part-8.
#        No variable names, hyper-parameters, or layers have been changed.

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import argparse
from torch.cuda.amp import autocast, GradScaler

# --------------------------------------------------------------------------------
# 1.  DEVICE & CLI ARGUMENTS
# --------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Part-11: Zero-shot VQA Evaluation")
parser.add_argument("--mode", type=str, choices=["inference"], required=True,
                    help="Only inference is supported in part-11")
parser.add_argument("--dataset", type=str, required=True,
                    help="Root of CLEVR dataset (containing images/ and questions/)")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the trained Part-8 checkpoint (.pt)")
parser.add_argument("--save_path", type=str, default=None,
                    help="Directory to save metrics & visualisations (default: ./part11_results)")
parser.add_argument("--variant", type=str, choices=["A", "B"], default="B",
                    help="Evaluate on variant A (seen) or B (zero-shot transfer)")
args = parser.parse_args()

# --------------------------------------------------------------------------------
# 2.  CONSTANTS  (UNMODIFIED FROM PART-8)
# --------------------------------------------------------------------------------
EMBED_DIM   = 768
MAX_LEN     = 30
BATCH_SIZE  = 128
NUM_EPOCHS  = 5
LEARNING_RATE = 5e-5

# --------------------------------------------------------------------------------
# 3.  DATASET (UNCHANGED)
# --------------------------------------------------------------------------------
class CLEVRDataset(Dataset):
    def __init__(self, root_dir, questions_file, split, transform=None,
                 max_len=MAX_LEN, tokenizer=None):
        self.root_dir  = root_dir
        self.split     = split
        self.transform = transform
        self.max_len   = max_len
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')

        with open(questions_file, 'r') as f:
            data = json.load(f)
            self.questions = data['questions']

        self.answer_to_idx, self.idx_to_answer = {}, {}
        answer_counts = {}
        for q in self.questions:
            if 'answer' in q:
                answer_counts[q['answer']] = answer_counts.get(q['answer'], 0) + 1

        sorted_answers = sorted(answer_counts.keys(),
                                key=lambda x: (-answer_counts[x], x))
        for idx, ans in enumerate(sorted_answers):
            self.answer_to_idx[ans] = idx
            self.idx_to_answer[idx] = ans

        self.num_classes = len(self.answer_to_idx)
        print(f"[{split}] Loaded {len(self.questions)} Qs, {self.num_classes} answers")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        img_path = os.path.join(self.root_dir, 'images', self.split, q['image_filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)

        enc = self.tokenizer(q['question'],
                             padding='max_length', truncation=True,
                             max_length=self.max_len, return_tensors='pt')
        item = {
            'image'         : image,
            'question'      : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'question_text' : q['question']
        }
        if 'answer' in q:
            item['answer'] = q['answer']
            item['answer_idx'] = torch.tensor(self.answer_to_idx[q['answer']],
                                              dtype=torch.long)
        return item

# --------------------------------------------------------------------------------
# 4.  MODEL  (IDENTICAL TO PART-8)
# --------------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        for i, child in enumerate(self.resnet_features.children()):
            if i < 6:
                for p in child.parameters(): p.requires_grad = False
        self.linear_projection = nn.Linear(2048, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        f = self.resnet_features(x)                       # [B,2048,h,w]
        b,c,h,w = f.size()
        f = f.view(b, c, h*w).permute(0,2,1)              # [B,h*w,2048]
        f = self.dropout(self.layer_norm(self.linear_projection(f)))
        return f                                          # [B,h*w,768]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, max_len=MAX_LEN,
                 num_layers=6, num_heads=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len+1, embed_dim))
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, num_heads,
                                               dim_feedforward=embed_dim*4,
                                               dropout=0.1, activation='gelu',
                                               batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers)

    def forward(self, x, attention_mask=None):
        b = x.size(0)
        tok_emb = self.token_embedding(x)
        cls_tok = self.cls_token.expand(b, -1, -1)
        emb = torch.cat([cls_tok, tok_emb], 1)
        emb = emb + self.positional_embedding[:, :emb.size(1), :]
        emb = self.dropout(self.layer_norm_1(emb))

        if attention_mask is not None:
            cls_mask = torch.ones((b,1), device=attention_mask.device)
            pad_mask = (1 - torch.cat([cls_mask, attention_mask], 1)).bool()
            return self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        return self.transformer_encoder(emb)

# --- REPLACE the whole CrossAttention class in part11.py with this block ----
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=8):
        super(CrossAttention, self).__init__()

        # Pre-layer norms
        self.layer_norm_q = nn.LayerNorm(embed_dim)
        self.layer_norm_k = nn.LayerNorm(embed_dim)

        # >>> original variable name: multihead_attn  (do NOT rename)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # >>> original variable name: ff_layer  (do NOT rename)
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        self.layer_norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, text_features, image_features):
        # CLS token from text
        cls_token = text_features[:, 0:1, :]

        # Normalisation before attention
        q = self.layer_norm_q(cls_token)
        k = self.layer_norm_k(image_features)

        # Cross-attention
        attn_out, _ = self.multihead_attn(query=q, key=k, value=k)

        # Residual -- CLS + attention
        fused = cls_token + attn_out

        # Feed-forward with residual
        fused = fused + self.ff_layer(self.layer_norm_ff(fused))

        # Return [B, 768]
        return fused.squeeze(1)
# ---------------------------------------------------------------------------


class Classifier(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=500, num_classes=28):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(self.dropout(x))

class VQAModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM):
        super().__init__()
        self.image_encoder  = ImageEncoder(embed_dim)
        self.text_encoder   = TextEncoder(vocab_size, embed_dim)
        self.cross_attention= CrossAttention(embed_dim)
        self.classifier     = Classifier(embed_dim, hidden_dim=500,
                                         num_classes=num_classes)

    def forward(self, img, q, attn_mask=None):
        img_f = self.image_encoder(img)
        txt_f = self.text_encoder(q, attn_mask)
        fused = self.cross_attention(txt_f, img_f)
        return self.classifier(fused)

# --------------------------------------------------------------------------------
# 5.  EVALUATION UTILITIES  (UNCHANGED EXCEPT FOR SMALL I/O TWEAKS)
# --------------------------------------------------------------------------------
def evaluate_model(model, loader, criterion=None):
    model.eval()
    all_preds, all_tgts = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            img  = batch['image'].to(device)
            q    = batch['question'].to(device)
            mask = batch['attention_mask'].to(device)
            out  = model(img, q, mask)
            preds = torch.argmax(out, 1)
            all_preds.extend(preds.cpu().numpy())

            if 'answer_idx' in batch:
                tgt = batch['answer_idx'].to(device)
                all_tgts.extend(tgt.cpu().numpy())
                if criterion:
                    loss_sum += criterion(out, tgt).item() * img.size(0)

    metrics = {}
    if all_tgts:
        metrics['accuracy'] = accuracy_score(all_tgts, all_preds)
        p,r,f1,_ = precision_recall_fscore_support(all_tgts, all_preds,
                                                   average='weighted',
                                                   zero_division=0)
        metrics.update({'precision':p, 'recall':r, 'f1_score':f1})
        if criterion:
            metrics['loss'] = loss_sum / len(loader.dataset)
    return metrics, all_preds

def visualize_samples(model, ds, idx_to_ans, indices, fname):
    model.eval()
    rows = len(indices)
    fig, axes = plt.subplots(rows, 1, figsize=(10,5*rows))
    if rows == 1: axes = [axes]
    for ax, idx in zip(axes, indices):
        d = ds[idx]
        img = d['image'].unsqueeze(0).to(device)
        q   = d['question'].unsqueeze(0).to(device)
        msk = d['attention_mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(img,q,msk),1).item()
        pred_ans = idx_to_ans[pred]
        np_img = img.squeeze(0).cpu().numpy().transpose(1,2,0)
        np_img = np.clip(np_img * np.array([0.229,0.224,0.225]) +
                         np.array([0.485,0.456,0.406]), 0, 1)
        ax.imshow(np_img)
        title = f"Q: {d['question_text']}\nPred: {pred_ans}"
        if 'answer' in d: title += f" | GT: {d['answer']}"
        ax.set_title(title); ax.axis('off')
    plt.tight_layout(); plt.savefig(fname); plt.close()

def save_metrics(metrics, extra, out_dir):
    path = os.path.join(out_dir, 'part11_metrics.txt')
    with open(path, 'w') as f:
        f.write("# Zero-shot Evaluation Metrics\n")
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n# Extra Info\n")
        for k,v in extra.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved metrics to {path}")

# --------------------------------------------------------------------------------
# 6.  MAIN:  ZERO-SHOT INFERENCE
# --------------------------------------------------------------------------------
def main():
    assert args.mode == 'inference', "Part-11 only supports inference mode."

    # ---- directories ----------------------------------------------------------
    out_dir = args.save_path or "./part11_results"
    os.makedirs(out_dir, exist_ok=True)

    # ---- tokenizer ------------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ---- rebuild answer mapping from trainA (as in Part-8 inference) ----------
    train_q_path = os.path.join(args.dataset, 'questions',
                                'CLEVR_trainA_questions.json')
    with open(train_q_path, 'r') as f:
        train_qs = json.load(f)['questions']
    ans_counts = {}
    for q in train_qs:
        ans_counts[q['answer']] = ans_counts.get(q['answer'], 0) + 1
    sorted_ans = sorted(ans_counts, key=lambda x: (-ans_counts[x], x))
    answer_to_idx = {a:i for i,a in enumerate(sorted_ans)}
    idx_to_answer = {i:a for a,i in answer_to_idx.items()}

    # ---- dataset & dataloader -------------------------------------------------
    variant = args.variant.upper()
    split = f"test{variant}"
    q_file = os.path.join(args.dataset, 'questions',
                          f"CLEVR_{split}_questions.json")
    if not os.path.isfile(q_file):
        raise FileNotFoundError(f"Questions file not found: {q_file}")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])

    test_ds = CLEVRDataset(args.dataset, q_file, split,
                           transform=transform, tokenizer=tokenizer)
    bs = BATCH_SIZE if torch.cuda.is_available() else min(8,BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=4 if torch.cuda.is_available() else 0,
                             pin_memory=torch.cuda.is_available())

    # ---- model ---------------------------------------------------------------
    model = VQAModel(tokenizer.vocab_size, len(answer_to_idx),
                     embed_dim=EMBED_DIM).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded checkpoint:", args.model_path)

    # ---- evaluation ----------------------------------------------------------
    crit = nn.CrossEntropyLoss()
    metrics, preds = evaluate_model(model, test_loader, crit)
    print("\nZero-shot Metrics:")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- visuals -------------------------------------------------------------
    sample_idx = list(range(5))
    vis_path = os.path.join(out_dir, f'sample_preds_{variant}.png')
    visualize_samples(model, test_ds, idx_to_answer, sample_idx, vis_path)
    print(f"Saved sample visualisations to {vis_path}")

    # ---- save metrics --------------------------------------------------------
    save_metrics(metrics, {'variant':variant, 'checkpoint':args.model_path}, out_dir)
    print(f"Done. Outputs are in: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()

