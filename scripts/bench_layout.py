"""Micro-benchmark 1 bước train ResNet-18 (batch 32, dữ liệu giả trên GPU): fp32 / AMP / channels_last / cudnn.benchmark.
Chạy: python -X utf8 scripts/bench_layout.py -> outputs/baseline/speed_layout_microbench.csv"""
import time, torch, torch.nn as nn, subprocess
from torchvision import models
dev='cuda'
def clk(): return subprocess.run("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader",shell=True,capture_output=True,text=True).stdout.strip()
def run(amp, cl, bench, n=60):
    torch.backends.cudnn.benchmark=bench
    m=models.resnet18(num_classes=7).to(dev)
    if cl: m=m.to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(m.parameters(),1e-4); sc=torch.amp.GradScaler('cuda',enabled=amp)
    x=torch.randn(32,3,224,224,device=dev); y=torch.randint(0,7,(32,),device=dev)
    if cl: x=x.contiguous(memory_format=torch.channels_last)
    for i in range(n+10):
        if i==10: torch.cuda.synchronize(); t=time.time()
        opt.zero_grad()
        with torch.autocast('cuda',torch.float16,enabled=amp): l=nn.functional.cross_entropy(m(x),y)
        sc.scale(l).backward(); sc.step(opt); sc.update()
    torch.cuda.synchronize(); dt=(time.time()-t)/n
    del m, opt
    print(f"amp={amp} cl={cl} bench={bench}: {dt*1000:.0f} ms/step | {clk()}",flush=True)
    return dt*1000
import csv, sys
rows=[]
for cfg in [(False,False,False),(True,False,True),(True,True,True),(False,False,True)]:
    rows.append(dict(zip(["amp","channels_last","cudnn_benchmark"],cfg), ms_per_step=run(*cfg)))
import pandas as pd; pd.DataFrame(rows).to_csv("outputs/baseline/speed_layout_microbench.csv", index=False); print(pd.DataFrame(rows))
