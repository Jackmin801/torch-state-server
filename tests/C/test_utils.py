import torch
from torchstate.C.utils import copy_bytes_to_tensor

def test_copy_bytes_to_tensor():
    tensor = torch.zeros(10)
    bytes_data = ('abcd' * 10).encode('utf-8')

    copy_bytes_to_tensor(tensor, bytes_data)

    assert list(tensor.untyped_storage()) == [97, 98, 99, 100] * 10
