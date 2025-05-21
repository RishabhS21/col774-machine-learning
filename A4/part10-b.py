import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import argparse
from torchvision.ops import sigmoid_focal_loss

from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Visual Question Answering Model")
parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--save_path", type=str)
parser.add_argument("--model_path", type=str)
args = parser.parse_args()

EMBED_DIM = 768
MAX_LEN = 30
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5

class CLEVRDataset(Dataset):
    def __init__(self, root_dir, questions_file, split, transform=None, max_len=MAX_LEN, tokenizer=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_len = max_len
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        with open(questions_file, 'r') as f:
            data = json.load(f)
            self.questions = data['questions']
            
        self.answer_to_idx = {}
        self.idx_to_answer = {}
        
        # First count frequencies to handle class imbalance better
        answer_counts = {}
        for q in self.questions:
            if 'answer' in q:
                answer = q['answer']
                if answer not in answer_counts:
                    answer_counts[answer] = 0
                answer_counts[answer] += 1
        
        # Sort by frequency to ensure consistent mapping across runs
        sorted_answers = sorted(answer_counts.keys(), key=lambda x: (-answer_counts[x], x))
        
        for idx, answer in enumerate(sorted_answers):
            self.answer_to_idx[answer] = idx
            self.idx_to_answer[idx] = answer
        
        self.num_classes = len(self.answer_to_idx)
        print(f"Loaded {len(self.questions)} questions with {self.num_classes} unique answers")
        
        # Print top 5 most common answers for verification
        print("Top 5 most common answers:")
        for answer in sorted_answers[:5]:
            print(f"  {answer}: {answer_counts[answer]} occurrences")
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question_data = self.questions[idx]
        question_text = question_data['question']
        image_filename = question_data['image_filename']
        
        img_path = os.path.join(self.root_dir, 'images', self.split, image_filename)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        encoding = self.tokenizer(
            question_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        question_tokens = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        if 'answer' in question_data:
            answer = question_data['answer']
            answer_idx = self.answer_to_idx[answer]
            return {
                'image': image,
                'question': question_tokens,
                'attention_mask': attention_mask,
                'answer_idx': torch.tensor(answer_idx, dtype=torch.long),
                'question_text': question_text,
                'answer': answer
            }
        else:
            return {
                'image': image,
                'question': question_tokens,
                'attention_mask': attention_mask,
                'question_text': question_text
            }

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        
        # Use more layers from ResNet for better feature extraction
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        
        
        self.linear_projection = nn.Linear(2048, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        features = self.resnet_features(x)
        
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        features = features.permute(0, 2, 1)
        
        features = self.linear_projection(features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        return features

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, max_len=MAX_LEN, num_layers=6, num_heads=8,pretrained_embeddings: torch.Tensor | None = None):
        super(TextEncoder, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            # copy BERT weights
            self.token_embedding.weight.data.copy_(pretrained_embeddings)
            # OPTIONAL: keep them frozen
            # self.token_embedding.weight.requires_grad = False
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len + 1, embed_dim))
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',  # Changed to GELU for better performance
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, attention_mask=None):
        batch_size = x.size(0)
        
        token_embeddings = self.token_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, token_embeddings], dim=1)
        embeddings = embeddings + self.positional_embedding[:, :embeddings.size(1), :]
        
        embeddings = self.layer_norm_1(embeddings)
        embeddings = self.dropout(embeddings)
        
        if attention_mask is not None:
            # Create mask for CLS token (always attend to it)
            cls_mask = torch.ones((batch_size, 1), device=attention_mask.device)
            extended_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            
            # Create key padding mask for transformer encoder
            # The transformer expects padding positions to be True, so invert the mask
            key_padding_mask = (1 - extended_attention_mask).bool()
            
            output = self.transformer_encoder(embeddings, src_key_padding_mask=key_padding_mask)
        else:
            output = self.transformer_encoder(embeddings)
        
        return output

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=8):
        super(CrossAttention, self).__init__()
        
        # Pre-layer normalization
        self.layer_norm_q = nn.LayerNorm(embed_dim)
        self.layer_norm_k = nn.LayerNorm(embed_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Add a feedforward layer for better fusion
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        
        self.layer_norm_ff = nn.LayerNorm(embed_dim)
    
    def forward(self, text_features, image_features):
        # Get [CLS] token from text features
        cls_token = text_features[:, 0, :].unsqueeze(1)
        
        # Apply layer normalization before attention
        cls_token_norm = self.layer_norm_q(cls_token)
        image_features_norm = self.layer_norm_k(image_features)
        
        # Cross attention: text [CLS] attends to image features
        attn_output, _ = self.multihead_attn(
            query=cls_token_norm,
            key=image_features_norm,
            value=image_features_norm
        )
        
        # Residual connection
        fused_features = cls_token + attn_output
        
        # Apply feed-forward layer with residual connection
        ff_output = self.ff_layer(self.layer_norm_ff(fused_features))
        fused_features = fused_features + ff_output
        
        # Final [CLS] token
        fused_features = fused_features.squeeze(1)
        
        return fused_features

class Classifier(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=500, num_classes=28):
        super(Classifier, self).__init__()
        
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.dropout(x)
        return self.mlp(x)

class VQAModel(nn.Module,pretrained_text_embeddings: torch.Tensor | None = None):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM):
        super(VQAModel, self).__init__()
        
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, pretrained_embeddings=pretrained_text_embeddings)
        self.cross_attention = CrossAttention(embed_dim)
        self.classifier = Classifier(embed_dim, hidden_dim=500, num_classes=num_classes)
    
    def forward(self, image, question, attention_mask=None):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(question, attention_mask)
        fused_features = self.cross_attention(text_features, image_features)
        logits = self.classifier(fused_features)
        
        return logits

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for epoch in range(num_epochs):
        model.train()
        train_loss = train_correct = train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = batch['image'].to(device, non_blocking=True)
            questions = batch['question'].to(device, non_blocking=True)
            attention_masks = batch['attention_mask'].to(device, non_blocking=True)
            answers = batch['answer_idx'].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images, questions, attention_masks)
                loss = criterion(outputs, answers)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == answers).sum().item()
            train_total += answers.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = batch['image'].to(device)
                questions = batch['question'].to(device)
                attention_masks = batch['attention_mask'].to(device)
                answers = batch['answer_idx'].to(device)

                outputs = model(images, questions, attention_masks)
                loss = criterion(outputs, answers)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == answers).sum().item()
                val_total += answers.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save checkpoint and log it to wandb
        ckpt_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': history
        }, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(save_path, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, best_path)
            print(f"Saved new best model at epoch {epoch+1} with val_acc={val_acc:.4f}")

    return history, best_val_acc


def evaluate_model(model, test_loader, criterion=None):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            
            outputs = model(images, questions, attention_masks)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            
            if 'answer_idx' in batch:
                answers = batch['answer_idx'].to(device)
                all_targets.extend(answers.cpu().numpy())
                
                if criterion:
                    loss = criterion(outputs, answers)
                    test_loss += loss.item() * images.size(0)
    
    metrics = {}
    if all_targets:
        metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        if criterion:
            metrics['loss'] = test_loss / len(test_loader.dataset)
    
    return metrics, all_predictions

def visualize_predictions(model, dataset, indices, idx_to_answer, filename='predictions.png'):
    model.eval()
    
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, 5 * len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        data = dataset[idx]
        image = data['image'].unsqueeze(0).to(device)
        question = data['question'].unsqueeze(0).to(device)
        attention_mask = data['attention_mask'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image, question, attention_mask)
            _, predicted = torch.max(outputs, 1)
            predicted_answer = idx_to_answer[predicted.item()]
        
        img = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Question: {data['question_text']}\nPredicted: {predicted_answer}")
        if 'answer' in data:
            axes[i].set_xlabel(f"Ground Truth: {data['answer']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_error_cases(model, dataset, idx_to_answer, num_cases=5, filename='error_cases.png'):
    model.eval()
    
    error_indices = []
    for idx in tqdm(range(min(len(dataset), 1000)), desc="Finding error cases"):
        if len(error_indices) >= num_cases:
            break
        
        data = dataset[idx]
        if 'answer' not in data:
            continue
        
        image = data['image'].unsqueeze(0).to(device)
        question = data['question'].unsqueeze(0).to(device)
        attention_mask = data['attention_mask'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image, question, attention_mask)
            _, predicted = torch.max(outputs, 1)
            predicted_answer = idx_to_answer[predicted.item()]
        
        if predicted_answer != data['answer']:
            error_indices.append(idx)
    
    if error_indices:
        visualize_predictions(model, dataset, error_indices, idx_to_answer, filename)
        print(f"Found {len(error_indices)} error cases. Visualized in '{filename}'")
    else:
        print("No error cases found in the first samples checked.")

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    history_path = os.path.join(save_path, 'training_history.png')
    plt.savefig(history_path)
    plt.close()
    print(f"Training history saved to {history_path}")
    
    np.save(os.path.join(save_path, 'history.npy'), history)

def save_metrics_report(metrics, model_summary, save_path):
    report_path = os.path.join(save_path, 'eval_metrics.txt')
    with open(report_path, 'w') as f:
        f.write("# Evaluation Metrics\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n# Model Summary\n\n")
        for key, value in model_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Evaluation metrics saved to {report_path}")

def focal_loss_sigmoid(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    logits: Tensor of shape [B, C]
    targets: LongTensor of shape [B] with values in [0..C-1]
    """
    # convert to one-hot [B, C]
    targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).float()
    return sigmoid_focal_loss(
        logits, 
        targets_onehot, 
        alpha=alpha, 
        gamma=gamma, 
        reduction=reduction
    )

def main():
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenizer   = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model  = BertModel.from_pretrained('bert-base-uncased')
    bert_embeds = bert_model.embeddings.word_embeddings.weight           # [vocab, 768]


    print(f"Running in {args.mode} mode")
    print(f"Dataset path: {args.dataset}")
    print(f"Save path: {args.save_path}")
    
    if args.mode == 'train':
        train_dataset = CLEVRDataset(
            root_dir=args.dataset,
            questions_file=os.path.join(args.dataset, 'questions', 'CLEVR_trainA_questions.json'),
            split='trainA',
            transform=image_transform,
            tokenizer=tokenizer
        )
        
        val_dataset = CLEVRDataset(
            root_dir=args.dataset,
            questions_file=os.path.join(args.dataset, 'questions', 'CLEVR_valA_questions.json'),
            split='valA',
            transform=image_transform,
            tokenizer=tokenizer
        )
        
        batch_size = BATCH_SIZE
        if not torch.cuda.is_available():
            batch_size = min(8, BATCH_SIZE)
            print(f"Using reduced batch size {batch_size} for CPU training")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        model = VQAModel(
            vocab_size=tokenizer.vocab_size,
            num_classes=train_dataset.num_classes,
            embed_dim=EMBED_DIM,
            pretrained_text_embeddings=bert_embeds
        ).to(device)
        
        # Use a weighted loss to handle class imbalance
        criterion = focal_loss_sigmoid
        
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Use AdamW with weight decay for better regularization
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        history, best_val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            save_path=args.save_path
        )
        
        plot_history(history, args.save_path)
        
        checkpoint = torch.load(os.path.join(args.save_path, 'best_model.pt'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_dataset = CLEVRDataset(
            root_dir=args.dataset,
            questions_file=os.path.join(args.dataset, 'questions', 'CLEVR_testA_questions.json'),
            split='testA',
            transform=image_transform,
            tokenizer=tokenizer
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        metrics, _ = evaluate_model(model, test_loader, criterion)
        print("\nTest Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        indices = list(range(5))
        visualize_predictions(model, test_dataset, indices, checkpoint['idx_to_answer'], 
                             os.path.join(args.save_path, 'sample_predictions.png'))
        
        visualize_error_cases(model, val_dataset, checkpoint['idx_to_answer'], 
                             filename=os.path.join(args.save_path, 'error_cases.png'))
        
        train_acc = history['train_acc'][-1]
        val_acc = history['val_acc'][-1]
        acc_diff = train_acc - val_acc
        
        model_summary = {
            'best_validation_accuracy': best_val_acc,
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1_score': metrics['f1_score'],
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'acc_difference': acc_diff
        }
        
        if acc_diff > 0.1:
            model_summary['model_status'] = "Overfitting (train acc significantly higher than val acc)"
        elif val_acc < 0.5:
            model_summary['model_status'] = "Underfitting (low validation accuracy)"
        else:
            model_summary['model_status'] = "Good balance between training and validation metrics"
        
        save_metrics_report(metrics, model_summary, args.save_path)
        
    elif args.mode == 'inference':
        # 1) Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)

        # 2) Reconstruct idx_to_answer from the training questions JSON
        train_qpath = os.path.join(args.dataset, 'questions', 'CLEVR_trainA_questions.json')
        with open(train_qpath, 'r') as f:
            train_questions = json.load(f)['questions']

        # Count answer frequencies
        answer_counts = {}
        for q in train_questions:
            if 'answer' in q:
                answer = q['answer']
                answer_counts[answer] = answer_counts.get(answer, 0) + 1

        # Sort by freq (desc) then lexicographically
        sorted_answers = sorted(
            answer_counts.keys(),
            key=lambda x: (-answer_counts[x], x)
        )

        # Build mappings
        answer_to_idx = {ans: idx for idx, ans in enumerate(sorted_answers)}
        idx_to_answer = {idx: ans for ans, idx in answer_to_idx.items()}

        # 3) Prepare dataset & model
        test_dataset = CLEVRDataset(
            root_dir=args.dataset,
            questions_file=os.path.join(args.dataset, 'questions', 'CLEVR_testA_questions.json'),
            split='testA',
            transform=image_transform,
            tokenizer=tokenizer
        )

        model = VQAModel(
            vocab_size=tokenizer.vocab_size,
            num_classes=len(idx_to_answer),
            embed_dim=EMBED_DIM,
            pretrained_text_embeddings=bert_embeds
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 4) DataLoader
        batch_size = BATCH_SIZE if torch.cuda.is_available() else min(8, BATCH_SIZE)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )

        # 5) Ensure writable output dir
        save_dir = args.save_path or os.path.join(os.getcwd(), 'inference_results')
        os.makedirs(save_dir, exist_ok=True)

        # 6) Run evaluation
        criterion = nn.CrossEntropyLoss()
        metrics, predictions = evaluate_model(model, test_loader, criterion)

        print("\nInference Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # 7) Visualize some samples and errors
        sample_indices = list(range(5))
        visualize_predictions(
            model, test_dataset, sample_indices, idx_to_answer,
            filename=os.path.join(save_dir, 'inference_predictions.png')
        )
        visualize_error_cases(
            model, test_dataset, idx_to_answer,
            filename=os.path.join(save_dir, 'inference_errors.png')
        )

        # 8) Save a text report
        save_metrics_report(metrics, {'model_path': args.model_path}, save_dir)

        print(f"\nInference complete. Outputs are in: {save_dir}")

main()
