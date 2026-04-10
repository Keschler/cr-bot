import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
from sklearn.metrics import confusion_matrix

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HAND_CLASSIFIER_PATH = MODELS_DIR / "hand_classifier_best.pt"


def main():
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        transforms.ToTensor(),
        transforms.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
          ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
          ),
    ])


    train_dataset = datasets.ImageFolder(
        "data/card_classifier/imagefolder/hand/train",
        transform=train_tf,
        allow_empty=True,
    )
    val_dataset = datasets.ImageFolder(
        "data/card_classifier/imagefolder/hand/val",
        transform=eval_tf,
        allow_empty=True,
    )

    num_classes = len(train_dataset.classes)

    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epochs = 20
    
    best_val_acc = 0


    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0 
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0 
        val_correct = 0
        val_total = 0 
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = loss_fn(logits, labels)


                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_dataset.classes,
                "val_acc": val_acc,
                "epoch": epoch + 1
                },
                       HAND_CLASSIFIER_PATH
                )
        print(
                  f"epoch {epoch+1}/{epochs} "
                  f"train_loss={train_loss/train_total:.4f} "
                  f"train_acc={train_correct/train_total:.4f} "
                  f"val_loss={val_loss/val_total:.4f} "
                  f"val_acc={val_correct/val_total:.4f}"
              )
    checkpoint = torch.load(HAND_CLASSIFIER_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    for i in range(len(cm)):
        cm[i, i] = 0 

    pairs = []
    for i in range(len(cm)):
        for l in range(len(cm)):
            if cm[i, l] > 0:
                pairs.append((cm[i, l], train_dataset.classes[i], train_dataset.classes[l]))
    pairs.sort(reverse=True)

    for count, true_name, pred_name in pairs[:20]:
        print(f"{true_name} -> {pred_name}: {count}")

if __name__ == "__main__":
    main()
