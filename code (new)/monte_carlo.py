import numpy as np
import matplotlib.pyplot as plt
import torch
from model import (
    prepare_data,
    DenseLSTMFusion,
    device,
    model_path
)

# load same data split
Xm_train, Xm_test, XM_train, XM_test, y_train, y_test = prepare_data()

# load trained model
model = DenseLSTMFusion(
    micro_dim=Xm_test.shape[2],
    macro_dim=XM_test.shape[2]
).to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

# monte carlo dropout
model.train()   # keep dropout ON

xm, xM = torch.tensor(Xm_test).to(device), torch.tensor(XM_test).to(device)

runs = 50

all_preds = []
with torch.no_grad():
    for _ in range(runs): 
        _, _, final_out = model(xm, xM)                  

        pred = final_out.cpu().numpy().flatten()

        all_preds.append(pred)

all_preds = np.array(all_preds)

# uncertainty
mean_pred = all_preds.mean(axis=0)
std_pred = all_preds.std(axis=0)

# print first 10
print('\nFirst 10 Predictions:\n')

for i in range(10):
    print(
        f'Sample {i+1}: '
        f'Future Score={mean_pred[i]:.3f} '
        f'± {std_pred[i]:.3f}'
    )

# plot
plt.figure(figsize=(14, 6))
plt.plot(mean_pred, label='Future Structural Score')
plt.fill_between(
    range(len(mean_pred)),
    mean_pred - std_pred,
    mean_pred + std_pred,
    alpha=0.3,
    label='Uncertainty Band'
)
plt.axhline(np.mean(mean_pred), linestyle='--')
plt.title('Monte Carlo Uncertainty')
plt.xlabel('Samples')
plt.ylabel('Score')
plt.legend()
plt.show()