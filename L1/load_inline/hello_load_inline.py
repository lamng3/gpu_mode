# %%capture
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install ninja

import torch
import os
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

build_dir = './tmp'
os.makedirs(build_dir, exist_ok=True)

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True,
    build_directory=build_dir
)

print(my_module.hello_world())