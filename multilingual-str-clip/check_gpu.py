"""
check_gpu.py
Standalone environment check: confirms whether PyTorch can see and use a
GPU on this machine, independent of any other project script. Run this
first whenever you're not sure if training will actually run on GPU.

Usage:
    python check_gpu.py
"""
import re
import sys
import warnings


def main():
    print("=" * 60)
    print(" GPU / ENVIRONMENT CHECK")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("!! PyTorch is not installed in this environment.")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"PyTorch version : {torch.__version__}")
    print(f"Python version  : {sys.version.split()[0]}")

    # Capture PyTorch's own CUDA-init warning rather than letting it print
    # to stderr and get lost — it contains the single most useful piece of
    # info when this fails (driver-too-old for this build), so we parse it
    # instead of only offering the generic "no GPU found" checklist.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cuda_available = torch.cuda.is_available()
    driver_warning = next((str(w.message) for w in caught if "driver" in str(w.message).lower()), None)

    print(f"\nCUDA available  : {cuda_available}")

    if cuda_available:
        print(f"CUDA version (PyTorch built against): {torch.version.cuda}")
        n = torch.cuda.device_count()
        print(f"GPU count       : {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"  [{i}] {props.name}  —  {total_gb:.1f} GB total, "
                  f"compute capability {props.major}.{props.minor}")

        current = torch.cuda.current_device()
        print(f"Current device  : cuda:{current} ({torch.cuda.get_device_name(current)})")

        # Actually exercise the GPU, not just query it — catches cases
        # where CUDA "is available" per the driver but something's
        # misconfigured (mismatched CUDA/driver versions, etc.) that only
        # shows up when a real op is run.
        try:
            x = torch.randn(1024, 1024, device="cuda")
            y = torch.randn(1024, 1024, device="cuda")
            torch.cuda.synchronize()
            (x @ y).sum().item()
            print("Matmul test     : OK — ran a real operation on the GPU successfully.")
        except Exception as e:
            print(f"Matmul test     : FAILED — {e}")
            print("  CUDA reports available but a real op failed; check driver/CUDA toolkit "
                  "versions match what this PyTorch build expects.")

        # TF32 status, since train.py/evaluate.py/inference.py enable this
        # automatically on CUDA.
        print(f"\nTF32 matmul allowed : {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN available     : {torch.backends.cudnn.is_available()}")
        if torch.backends.cudnn.is_available():
            print(f"cuDNN version       : {torch.backends.cudnn.version()}")

        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"\nGPU memory allocated: {mem_alloc:.1f} MB")
        print(f"GPU memory reserved : {mem_reserved:.1f} MB")

    else:
        mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        print(f"MPS available (Apple Silicon GPU): {mps_available}")

        if driver_warning and "too old" in driver_warning.lower():
            # This is the specific, very common case: PyTorch was built
            # against a newer CUDA version than the installed NVIDIA
            # driver supports. Drivers are backward-compatible (a newer
            # driver runs older CUDA builds) but NOT forward-compatible
            # (an older driver can't run a build compiled for a newer
            # CUDA version than it supports) — so this needs a specific
            # fix, not the generic "no GPU" checklist.
            found_ver = re.search(r"found version (\d+)", driver_warning)
            if found_ver:
                raw = int(found_ver.group(1))
                # CUDA encodes versions as major*1000 + minor*10 (e.g. 12080 -> 12.8)
                driver_str = f"{raw // 1000}.{(raw % 1000) // 10}"
            else:
                driver_str = "?"
            built_cuda = torch.version.cuda or "unknown"
            print(f"\n!! DRIVER TOO OLD FOR THIS PYTORCH BUILD (not a general 'no GPU' issue):")
            print(f"   PyTorch {torch.__version__} was built for CUDA {built_cuda}, but your "
                  f"installed NVIDIA driver only supports up to CUDA {driver_str}.")
            print(f"   Drivers run older CUDA builds fine, but can't run a build compiled for a")
            print(f"   NEWER CUDA version than they support — that's what's happening here.")
            print(f"\n   Fix — pick ONE:")
            print(f"   A) Update your NVIDIA driver (needs to support CUDA {built_cuda}+):")
            print(f"      https://www.nvidia.com/Download/index.aspx")
            print(f"   B) (usually faster) Install a PyTorch build matching your CURRENT driver instead:")
            print(f"      pip uninstall torch torchvision")
            print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print(f"      (swap cu121 for whichever CUDA version your driver supports — check with")
            print(f"       `nvidia-smi`, top-right corner shows 'CUDA Version: X.Y')")
            print(f"\n   Raw warning PyTorch gave: {driver_warning.strip()}")
        else:
            print("\n!! No CUDA GPU detected. Common causes:")
            print("   - No NVIDIA GPU in this machine/VM")
            print("   - NVIDIA driver not installed — run `nvidia-smi` in a terminal to check")
            print("   - PyTorch was installed as the CPU-only build — reinstall the CUDA build "
                  "from https://pytorch.org/get-started/locally/ matching your CUDA version")
            print("   - Running inside a container/VM without GPU passthrough enabled")

        print("\n   All project scripts (train.py, evaluate.py, inference.py, sanity_check.py) "
              "will still run on CPU, just much slower — they auto-detect via "
              "`--device cuda if torch.cuda.is_available() else cpu`.")

    print("\n" + "=" * 60)
    print(f" VERDICT: {'GPU is ready to use.' if cuda_available else 'Running on CPU only.'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
