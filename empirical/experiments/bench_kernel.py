"""Benchmark: scff_signal kernel correctness + speed vs torch; SCFF-vs-BP peak memory vs depth.
Run: python experiments/bench_kernel.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda.scff_ext import scff_signal
from cuda.reference import signal_ref
from gpu_arch import ConvSCFF, _pooled, scff_local_step

def _time(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters

def bench_kernel():
    print("kernel correctness + speed (vs torch reference):")
    for B, C in [(64, 128), (128, 256), (256, 512)]:
        z = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        zp = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        s_k, _ = scff_signal(z, zp, 0.5); s_r, _ = signal_ref(z, zp, 0.5)
        err = float((s_k - s_r).abs().max())
        t_k = _time(lambda: scff_signal(z, zp, 0.5))
        t_r = _time(lambda: signal_ref(z, zp, 0.5))
        print(f"  B={B:4d} C={C:4d}  maxerr={err:.2e}  kernel={t_k:.4f}ms  torch={t_r:.4f}ms  "
              f"speedup={t_r/t_k:.2f}x", flush=True)

def bench_memory():
    print("\npeak memory vs depth (SCFF forward-only vs BP):")
    x = torch.randn(128, 3, 32, 32, device="cuda")
    xp = x + 0.1 * torch.randn_like(x)
    for L in [8, 16, 32, 64]:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        m = ConvSCFF(64, L, "residual", 1.0/math.sqrt(L)).cuda()
        scff_local_step(m, x, xp, 0.5, 0.05)
        scff_mb = torch.cuda.max_memory_allocated()/1e6
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        m = ConvSCFF(64, L, "residual", 1.0/math.sqrt(L)).cuda()
        head = torch.nn.Linear(64, 10).cuda()
        loss = torch.nn.functional.cross_entropy(head(m.pooled(m(x)[-1])),
                                                 torch.zeros(128, dtype=torch.long, device="cuda"))
        loss.backward()
        bp_mb = torch.cuda.max_memory_allocated()/1e6
        print(f"  L={L:3d}  SCFF={scff_mb:7.1f}MB  BP={bp_mb:7.1f}MB  ratio={bp_mb/scff_mb:.1f}x",
              flush=True)

def main():
    assert torch.cuda.is_available(), "GPU required"
    bench_kernel(); bench_memory()

if __name__ == "__main__":
    main()
