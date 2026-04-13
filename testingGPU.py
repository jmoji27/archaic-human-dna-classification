import torch
import time

print("=== BASIC INFO ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA version (torch):", torch.version.cuda)

print("\n=== DEVICE CHECK ===")
device = torch.device("cuda")

x = torch.rand(1000, 1000).to(device)
y = torch.rand(1000, 1000).to(device)

z = torch.mm(x, y)

print("Tensor device:", z.device)

print("\n=== PERFORMANCE TEST ===")

# CPU test
x_cpu = torch.rand(3000, 3000)
y_cpu = torch.rand(3000, 3000)

start = time.time()
z_cpu = torch.mm(x_cpu, y_cpu)
cpu_time = time.time() - start

# GPU test
x_gpu = x_cpu.to(device)
y_gpu = y_cpu.to(device)

torch.cuda.synchronize()
start = time.time()
z_gpu = torch.mm(x_gpu, y_gpu)
torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f} sec")
print(f"GPU time: {gpu_time:.4f} sec")

print("\n=== RESULT ===")
if gpu_time < cpu_time:
    print("GPU is being used correctly")
else:
    print("GPU not effectively used")
