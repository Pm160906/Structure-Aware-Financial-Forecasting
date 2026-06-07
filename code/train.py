import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from model import prepare_data, DenseLSTMFusion, device, model_path

# dataset
class MarketDataset(Dataset):
    def __init__(self, Xm, XM, y):
        self.Xm = torch.tensor(Xm)
        self.XM = torch.tensor(XM)
        self.y = torch.tensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.Xm[idx], self.XM[idx], self.y[idx]

# load data
Xm_train, Xm_test, XM_train, XM_test, y_train, y_test = prepare_data()

train_ds = MarketDataset(Xm_train, XM_train, y_train)
test_ds  = MarketDataset(Xm_test, XM_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# model
model = DenseLSTMFusion(
    micro_dim=Xm_train.shape[2],
    macro_dim=XM_train.shape[2]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

alpha = 0.5
epochs = 20
# training
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xm, xM, target in train_loader:
        xm, xM = xm.to(device), xM.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        micro_out, macro_out, final_out = model(xm, xM)

        micro_loss = criterion(micro_out, target)
        macro_loss = criterion(macro_out, target)
        final_loss = criterion(final_out, target)

        loss = (
            alpha * micro_loss +
            (1 - alpha) * macro_loss +
            final_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')

# evaluation
model.eval()

preds, truth = [], []
with torch.no_grad():
    for xm, xM, target in test_loader:
        xm, xM = xm.to(device), xM.to(device)
        _, _, final_out = model(xm, xM)

        preds.extend(final_out.cpu().numpy().flatten())
        truth.extend(target.numpy().flatten())


mse = mean_squared_error(truth, preds)
print("\nFinal Test MSE:", mse)

# save
torch.save(model.state_dict(), model_path)
print('Saved model.')