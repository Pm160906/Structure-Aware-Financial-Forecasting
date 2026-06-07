import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# 1. generate data
np.random.seed(0)
n = 200
crash_point = 120

trend = np.linspace(50, 100, n)
noise = np.random.normal(0, 2, n)
trend[crash_point:] -= np.linspace(0, 60, n - crash_point)

# add chaos AFTER crash
noise[crash_point:] += np.random.normal(0, 5, n - crash_point)
data = trend + noise

# 2. hankel matrix
def hankel(series, window):
    return np.array([series[i:i+window] for i in range(len(series)-window)])

# 3. structure score (svd)
def structure_score(segment):
    H = hankel(segment, 8)
    _, S, _ = np.linalg.svd(H, full_matrices=False)
    return S[0] / np.sum(S)

# 4. sliding window
window = 20
scores = []

for i in range(len(data) - window):
    segment = data[i:i+window]
    scores.append(structure_score(segment))

scores = np.array(scores)

# 5. optimization (unit 2) - learn threshold using gradient descent
threshold = 0.9
lr = 0.01

labels = np.array([0 if i < crash_point else 1 for i in range(len(scores))])

for _ in range(200):
    preds = (scores < threshold).astype(int)
    loss = np.mean((preds - labels)**2)
    
    # gradient approximation
    grad = np.mean(2*(preds - labels))
    threshold -= lr * grad

# 6. statistical test (unit 3)
before = scores[:100]
after = scores[120:]

t_stat, p_val = ttest_ind(before, after)

print("T-stat:", t_stat)
print("P-value:", p_val)

# 7. plots
plt.figure()
plt.plot(data)
plt.axvline(x=crash_point, linestyle='--')
plt.title("Stock Price")

plt.figure()
plt.plot(scores)
plt.axvline(x=crash_point, linestyle='--')
plt.title("Structure Score")

plt.show()