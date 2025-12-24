import torch

print(f"PyTorch version: {torch.__version__}")

# check mps backend for M1 Macbooks
if torch.backends.mps.is_available():
    print("✅ MPS backend available")
    device = torch.device("mps")
else:
    print("❌ MPS backend not available")
    device = torch.device("cpu")

# test tensor operation
x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x @ y

print(f"Test successful! Device: {device}, Result shape: {z.shape}")

