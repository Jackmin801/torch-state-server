from typing import Optional, Any, TypeVar, Type
import torch
import socket
import struct
from torchstate.C.utils import copy_bytes_to_tensor
from torchstate.ttype_consts import TransferType, ScalarTransferType, TTYPE_TO_ELEMENT_SIZE, TTYPE_TO_CODEBOOK_SIZE

T = TypeVar('T')

class StateClientError(Exception):
    pass

SCALAR_TYPE_MAPPING = {
    ScalarTransferType.FLOAT64: (8, 'd', float),
    ScalarTransferType.INT64: (8, 'q', int),
    ScalarTransferType.BOOL8: (1, '?', bool),
    ScalarTransferType.STR: (None, None, str),  # Special case handled separately
}

CHUNK_SIZE = 2 * 4096

def recv_exact(sock: socket.socket, size: int, chunk_size: int = CHUNK_SIZE) -> bytes:
    """Receive exactly size bytes from socket, handling large transfers in chunks."""
    data = bytearray()
    remaining = size
    
    while remaining > 0:
        chunk = min(remaining, chunk_size)
        received = sock.recv(chunk)
        if not received:
            raise ConnectionError("Connection closed before receiving all data")
        data.extend(received)
        remaining -= len(received)
    
    return bytes(data)

def _pack_request(path: str, transfer_type: int, size: int) -> bytes:
    if len(path) > 244:
        raise ValueError("Path length must be at most 244 characters")
    encoded_path = path.encode().ljust(244, b'\x00')
    return struct.pack('244siq', encoded_path, transfer_type, size)

class StateClient:
    def __init__(self, url: str):
        self.hostname, port = url.split(":")
        self.port = int(port)
        #self._init_socket()

    def _init_socket(self):
        """Initialize a new socket connection"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.hostname, self.port))

    def _handle_error_response(self, succ: int, ttype: int, size: int):
        """Handle error responses from the server. Raises an exception if not successful"""
        if succ != 0:
            if ttype == ScalarTransferType.STR.value:
                response = self.client_socket.recv(size)
                self.client_socket.close()
                raise StateClientError(response.decode())
            self.client_socket.close()
            raise StateClientError("Failed to get value")

    def get_tensor(
        self,
        path: str,
        transfer_type: Optional[TransferType] = None,
        inplace_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pack the request
        encoded_transfer_type = transfer_type.value if transfer_type else -1
        encoded_size = inplace_tensor.numel() if inplace_tensor is None else -1
        packed_request = _pack_request(path, encoded_transfer_type, encoded_size)

        # Reset socket connection
        self._init_socket()

        try:
            # Send packed request
            self.client_socket.sendall(packed_request)

            # Unpack the header
            if inplace_tensor is None:
                tensor_meta = self.client_socket.recv(64)
                tensor_meta = struct.unpack('iiqiiiiiiiiiiii', tensor_meta)
                succ, ttype, size = tensor_meta[0], tensor_meta[1], tensor_meta[2]
                shapes = tuple(i for i in tensor_meta[3:9] if i != -1)
                stride = tuple(i for i in tensor_meta[9:15] if i != -1)

                inplace_tensor = torch.empty(size)
                inplace_tensor.as_strided_(shapes, stride)
            else:
                resp_header = self.client_socket.recv(16)
                succ, ttype, size = struct.unpack('iiq', resp_header)

            # Check for errors
            self._handle_error_response(succ, ttype, size)

            # Get Codebook if needed
            if ttype in TTYPE_TO_CODEBOOK_SIZE:
                codebook_size = TTYPE_TO_CODEBOOK_SIZE[ttype]
                _ = self.client_socket.recv(codebook_size)  # codebook currently unused

            # Receive the tensor data
            elem_size = TTYPE_TO_ELEMENT_SIZE[ttype]
            tensor_data = recv_exact(self.client_socket, elem_size * size)
            copy_bytes_to_tensor(inplace_tensor, tensor_data)

            return inplace_tensor

        finally:
            self.client_socket.close()

    def _get_scalar(self, path: str, scalar_type: ScalarTransferType, expected_type: Type[T]) -> T:
        """Generic method to handle scalar data retrieval"""
        size, fmt, _ = SCALAR_TYPE_MAPPING[scalar_type]
        
        # Special handling for strings
        if scalar_type == ScalarTransferType.STR:
            packed_request = _pack_request(path, scalar_type.value, -1)
        else:
            packed_request = _pack_request(path, scalar_type.value, size)

        # Reset socket connection
        self._init_socket()
        
        try:
            # Send request
            self.client_socket.sendall(packed_request)

            # Receive and parse header
            resp_header = self.client_socket.recv(16)
            succ, ttype, recv_size = struct.unpack('iiq', resp_header)
            
            # Check for errors
            self._handle_error_response(succ, ttype, recv_size)

            # Handle string separately
            if scalar_type == ScalarTransferType.STR:
                data = self.client_socket.recv(recv_size)
                return data.decode()

            # Handle other scalar types
            data = self.client_socket.recv(size)
            return struct.unpack(fmt, data)[0]

        finally:
            self.client_socket.close()

    def get_float(self, path: str) -> float:
        return self._get_scalar(path, ScalarTransferType.FLOAT64, float)

    def get_int(self, path: str) -> int:
        return self._get_scalar(path, ScalarTransferType.INT64, int)

    def get_str(self, path: str) -> str:
        return self._get_scalar(path, ScalarTransferType.STR, str)

    def get_bool(self, path: str) -> bool:
        return self._get_scalar(path, ScalarTransferType.BOOL8, bool)