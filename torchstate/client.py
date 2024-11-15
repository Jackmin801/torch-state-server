from typing import Optional
import torch
import socket
import struct
from torchstate.C.utils import copy_bytes_to_tensor
from enum import Enum

class StateClientError(Exception):
    pass

class ScalarTransferType(Enum):
    STR = 0
    INT64 = 1
    FLOAT64 = 2
    BOOL8 = 3

class TransferType(Enum):
    FLOAT32 = 4
    BFLOAT16 = 5
    FLOAT16 = 6
    UNIFORM_INT8 = 7

TTYPE_TO_CODEBOOK_SIZE = {
    TransferType.UNIFORM_INT8.value: 256,
}

TTYPE_TO_ELEMENT_SIZE = {
    TransferType.FLOAT32.value: 4,
    TransferType.BFLOAT16.value: 2,
    TransferType.FLOAT16.value: 2,
    TransferType.UNIFORM_INT8.value: 1,
}

def _pack_request(path: str, transfer_type: int, size: int) -> bytes:
    # Pack the request
    if len(path) > 244:
        raise ValueError("Path length must be at most 252 characters")
    encoded_path = path.encode().ljust(244, b'\x00')
    packed_request = struct.pack('244siq', encoded_path, transfer_type, size)
    return packed_request

class StateClient:
    def __init__(self, url: str):
        self.hostname, port = url.split(":")
        self.port = int(port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def get_tensor(
        self,
        path: str,
        transfer_type: Optional[TransferType] = None,
        inplace_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pack the request
        encoded_transfer_type = transfer_type.value if transfer_type else -1
        encoded_size = inplace_tensor.numel() if inplace_tensor else -1
        packed_request = _pack_request(path, encoded_transfer_type, encoded_size)

        # Send packed request to the server
        self.client_socket.connect((self.hostname, self.port))
        self.client_socket.sendall(packed_request)

        # Unpack the header
        if inplace_tensor is None:
            tensor_meta = self.client_socket.recv(64)
            tensor_meta = struct.unpack('iiqiiiiiiiiiiii', tensor_meta)
            succ, ttype, size = tensor_meta[0], tensor_meta[1], tensor_meta[2]
            shapes = (i for i in tensor_meta[3:3 + 6] if i != -1)
            stride = (i for i in tensor_meta[9:9 + 6] if i != -1)

            inplace_tensor = torch.empty(size)
            inplace_tensor.as_strided_(shapes, stride)
        else:
            resp_header = self.client_socket.recv(16)
            succ, ttype, size = struct.unpack('iiq', resp_header)
        
        # Server error response
        if succ != 0:
            if ttype == ScalarTransferType.STR.value:
                response = self.client_socket.recv(size)
                self.client_socket.close()
                raise StateClientError(response.decode())
            else:
                self.client_socket.close()
                raise StateClientError("Failed to get tensor")

        # Get Codebook
        if ttype in TTYPE_TO_CODEBOOK_SIZE:
            codebook_size = TTYPE_TO_CODEBOOK_SIZE[ttype]
            codebook = self.client_socket.recv(codebook_size)

        # Receive the tensor data
        elem_size = TTYPE_TO_ELEMENT_SIZE[ttype]
        tensor_data = self.client_socket.recv(elem_size * size)
        copy_bytes_to_tensor(inplace_tensor, tensor_data)

        # Close the client socket
        self.client_socket.close()
    
    def get_float(self, path: str) -> float:
        # Pack the request
        packed_request = _pack_request(path, ScalarTransferType.FLOAT64, 8)

        # Send packed request to the server
        self.client_socket.connect((self.hostname, self.port))
        self.client_socket.sendall(packed_request)

        # Unpack the header
        resp_header = self.client_socket.recv(16)
        succ, ttype, size = struct.unpack('iiq', resp_header)
    
        # Server error response
        if succ != 0:
            if ttype == ScalarTransferType.STR.value:
                response = self.client_socket.recv(size)
                raise StateClientError(response.decode())
            else:
                raise StateClientError("Failed to get float")
        
        # Receive the float data
        float_data = self.client_socket.recv(8)
        self.client_socket.close()
        return struct.unpack('d', float_data)[0]


    def get_int(self, path: str) -> int:
        pass

    def get_str(self, path: str) -> str:
        pass

    def get_bool(self, path: str) -> bool:
        pass
