import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from model import DenseLSTMFusion, device, model_path
from hankelization import prepare_hankel_data

# load test data
_, Xm_test, _, XM_test, _, y_test = prepare_hankel_data()

ds = TensorDataset(
    torch.tensor(Xm_test, dtype=torch.float32),
    torch.tensor(XM_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
)
loader = DataLoader(ds, batch_size=64, shuffle=False)

# load model
model = DenseLSTMFusion(
    micro_dim=Xm_test.shape[2],
    macro_dim=XM_test.shape[2]
).to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)
model.eval()

# predict
preds, truth = [], []
with torch.no_grad():
    for xm, xM, target in loader:
        xm, xM = xm.to(device), xM.to(device)                # move input tensors to CPU/GPU

        _, _, out = model(xm, xM)                            # model returns: micro_out, macro_out, final_out

        preds.extend(out.cpu().numpy().flatten())            # store final fused prediction
        truth.extend(target.numpy().flatten())               # store true target values

# convert lists into arrays
y_true = np.array(truth)
y_pred = np.array(preds)

# metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

actual_dir = np.sign(np.diff(y_true))
pred_dir = np.sign(np.diff(y_pred))
trend_acc = np.mean(actual_dir == pred_dir)

# print report
print('=' * 45)
print('Dense LSTM Fusion Evaluation')
print('=' * 45)
print(f'MSE: {mse:.6f}')
print(f'RMSE: {rmse:.6f}')
print(f'Trend Accuracy: {trend_acc:.4f}')
print('=' * 45)

# plots
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Left: Original data
ax[0].plot(y_true, label='Actual Values', linewidth=2)
ax[0].set_title('Original Stock Data')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Value')
ax[0].legend()
ax[0].grid(True)

# Right: Prediction vs Actual
ax[1].plot(y_true, label='Actual', linewidth=2)
ax[1].plot(y_pred, label='Predicted', linestyle='--')
ax[1].set_title('Model Prediction vs Actual')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Value')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

actual_dir = np.sign(np.diff(y_true))
pred_dir = np.sign(np.diff(y_pred))

acc = np.mean(actual_dir == pred_dir)
print('Trend Accuracy:', acc)