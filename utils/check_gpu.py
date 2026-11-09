"""This code is for to check the gpu, if it supports FP16 and BF16."""

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute Capability:", torch.cuda.get_device_capability(0))

    print("\nPrecision Support:")
    print("FP16:", torch.cuda.is_available())
    print("BF16:", torch.cuda.is_bf16_supported())

    print("\nTesting BF16 computation:")

    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)

        c = a @ b
        torch.cuda.synchronize()

        print("BF16 matrix multiplication: SUCCESS")
        print("Output dtype:", c.dtype)

    except Exception as e:
        print("BF16 test failed:", e)