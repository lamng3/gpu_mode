# %%capture
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# more complex with warm ups
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(prof):
  print(prof.key_averages().table(
      sort_by="cuda_time_total", row_limit=-1))
  prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

with torch.profiler.profile(
    activities=[
      torch.profiler.ProfilerActivity.CPU,
      torch.profiler.ProfilerActivity.CUDA  
    ],

    # in the example with wait=1, warmup=1, active=2, repeat=1
    # profiler will skip the 1st step/iteration
    # start warming up on the 2nd
    # record the 3rd and 4th iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule = torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    
    on_trace_ready=trace_handler
    # on_trace_read=torch.profiler.tensorboard_trace_handler('./log)
    # used when outputting for tensorboard
) as prof:
  for iter in range(10):
      torch.square(torch.randn(10000, 10000).cuda())
      # send a signal to the profiler that the next iteration has started
      prof.step()

