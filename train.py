import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import numpy as np

from data_loader import ESC50Dataset, load_train_val_indices
from model import AudioCNN

device = torch.device("cpu") 

full_ds = ESC50Dataset()
train_idx, val_idx = load_train_val_indices()
train_ds = Subset(full_ds, train_idx)
val_ds = Subset(full_ds, val_idx)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = AudioCNN(num_classes=50).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("🚀 Training 50-class ESC-50 model...")

EPOCHS = 30  

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)              
        loss = criterion(out, y)    
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))

    model.eval()
    all_acc = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            preds = out.argmax(dim=1)
            acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
            all_acc.append(acc)

    val_acc = float(np.mean(all_acc)) if all_acc else 0.0

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.3f}  ValAcc: {val_acc:.3f}")
output_path = "model_50class_retrained.pth"
torch.save(model.state_dict(), output_path)
print(f"✅ Saved {output_path}; the original model_50class.pth was left untouched.")
