# %%capture
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# easy and default way to use profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
  for _ in range(10):
    a = torch.square(torch.randn(10000, 10000).cuda())

prof.export_chrome_trace("pt_profiler_default_trace.json")