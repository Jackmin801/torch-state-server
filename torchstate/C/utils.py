from torch.utils.cpp_extension import load
from pathlib import Path
import torch

UTILS_CSRC_PATH = Path(__file__).parent / "csrc" / "utils.cpp"

_utils = load(
    name="utils",
    sources=[UTILS_CSRC_PATH],
    extra_cflags=['-O3'],
    verbose=False
)

def copy_bytes_to_tensor(tensor: torch.Tensor, bytes_data: bytes) -> None:
    _utils.copy_bytes_to_tensor(tensor, bytes_data)
