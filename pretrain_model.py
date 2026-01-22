from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import os

# =========================
# 1️⃣ 固定隨機種子
# =========================
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2️⃣ 讀取 CSV
# =========================
df = pd.read_csv(r"C:\Users\user\Desktop\rice_disease\rice_disease_label_df.csv")
num_classes = df['label_id'].nunique()

# =========================
# 3️⃣ Image Processor
# =========================
processor = AutoImageProcessor.from_pretrained(
    "cvmil/dinov2-base_rice-leaf-disease-augmented_fft"
)

# =========================
# 4️⃣ Dataset
# =========================
class RiceLeafCSV(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        label = row['label_id']

        image_path = os.path.join(self.image_dir, f"{fname}.JPG")
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, torch.tensor(label)

# =========================
# 5️⃣ Train / Validate function
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for inputs, labels in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# =========================
# 6️⃣ Five-Fold Cross Validation
# =========================
image_dir = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\影像檔"

n_splits = 5
num_epochs = 15
patience = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_id'])):
    print(f"\n========== Fold {fold+1}/{n_splits} ==========")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = RiceLeafCSV(train_df, image_dir, processor)
    val_dataset = RiceLeafCSV(val_df, image_dir, processor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 🔁 每個 fold 重新初始化模型
    model = AutoModelForImageClassification.from_pretrained(
        "cvmil/dinov2-base_rice-leaf-disease-augmented_fft"
    )

    model.classifier = nn.Linear(
        model.classifier.in_features, num_classes
    )

    # freeze backbone
    for param in model.dinov2.parameters():
        param.requires_grad = False

    for name, param in model.dinov2.named_parameters():
        if "encoder.layers.11" in name:
            param.requires_grad = True

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW([
        {"params": model.classifier.parameters(), "lr": 1e-3},
        {"params": [p for p in model.dinov2.parameters() if p.requires_grad], "lr": 5e-5}
    ])

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        print(
            f"Fold {fold+1} | Epoch {epoch+1} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                fr"C:\Users\user\Desktop\rice_disease\best_dinov2_fold_{fold+1}.pth"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    fold_accuracies.append(best_val_acc)

# =========================
# 7️⃣ Final Result
# =========================
fold_accuracies = np.array(fold_accuracies)
print("\n===== Five-Fold Cross Validation Result =====")
print(f"Mean Accuracy: {fold_accuracies.mean():.4f}")
print(f"Std  Accuracy: {fold_accuracies.std():.4f}")


# ---- 驗證部分 ---- 
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def validate_collect(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return running_loss / len(all_labels), acc, all_labels, all_preds

all_fold_labels = []
all_fold_preds = []
per_fold_reports = []

val_loss, val_acc, val_labels, val_preds = validate_collect(
    model, val_loader, criterion
)

if val_acc > best_val_acc:
    best_val_acc = val_acc
    epochs_no_improve = 0

    best_fold_labels = val_labels
    best_fold_preds = val_preds

    torch.save(
        model.state_dict(),
        fr"C:\Users\user\Desktop\rice_disease\best_dinov2_fold_{fold+1}.pth"
    )

all_fold_labels.extend(best_fold_labels)
all_fold_preds.extend(best_fold_preds)

report = classification_report(
    best_fold_labels,
    best_fold_preds,
    output_dict=True,
    zero_division=0
)
per_fold_reports.append(report)

cm = confusion_matrix(all_fold_labels, all_fold_preds)

cm_df = pd.DataFrame(cm)
cm_df.to_csv(
    r"C:\Users\user\Desktop\rice_disease\five_fold_confusion_matrix.csv",
    index=False
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Five-Fold Confusion Matrix")
plt.tight_layout()
plt.savefig(
    r"C:\Users\user\Desktop\rice_disease\five_fold_confusion_matrix.png",
    dpi=300
)
plt.close()

class_names = [str(i) for i in range(num_classes)]
metrics = ["precision", "recall", "f1-score"]

avg_report = {}

for cls in class_names:
    avg_report[cls] = {}
    for m in metrics:
        avg_report[cls][m] = np.mean([
            fold[cls][m] for fold in per_fold_reports
        ])

metrics_df = pd.DataFrame(avg_report).T
metrics_df.to_csv(
    r"C:\Users\user\Desktop\rice_disease\five_fold_per_class_metrics.csv"
)

