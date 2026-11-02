# %%
from pathlib import Path
import numpy as np
DATA_DIR = Path("data/instagram")
x_text = np.load(DATA_DIR / "x_text.npy")
y = np.load(DATA_DIR / "y.npy")
edge_index = np.load(DATA_DIR / "edge_index.npy")

print("X-text")

for i in range(5):
# %%
    print(f'X-text for {i}th')

    print(x_text[i])
print("Y-text")

# %%
print(y[:5])

print("Edge-index")
print(edge_index[:5])

print("Edge Transformned")
edge_index = edge_index.T          # (144010, 2) → (2, 144010), the shape PyG expects
print(edge_index[:5])
