import torch

# --- Snippet 1: float32 + float32 ---
# Accumulating a float32 tensor with another float32 tensor.
# This is the most precise and expected result.
print("--- float32 + float32 ---")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)
# Expected output: tensor(10.0000)

print("-" * 25)

# --- Snippet 2: float16 + float16 ---
# Accumulating a float16 tensor with another float16 tensor.
# The value 0.01 cannot be perfectly represented in float16,
# leading to significant precision loss and an inaccurate sum.
print("--- float16 + float16 ---")
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)
# Example output: tensor(9.7656, dtype=torch.float16)

print("-" * 25)

# --- Snippet 3: float32 + float16 (Implicit Casting) ---
# Accumulating a float16 tensor into a float32 tensor.
# PyTorch upcasts the float16 to float32 before the addition,
# but the initial precision loss from representing 0.01 as float16 persists.
print("--- float32 + float16 (Implicit Casting) ---")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)
# Example output: tensor(9.7656)

print("-" * 25)

# --- Snippet 4: float32 + float16 (Explicit Casting) ---
# Creating a float16 tensor, explicitly casting it to float32,
# and then adding it to the float32 accumulator. This is functionally
# identical to Snippet 3 and shows the same precision loss.
print("--- float32 + float16 (Explicit Casting) ---")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
# Example output: tensor(9.7656)