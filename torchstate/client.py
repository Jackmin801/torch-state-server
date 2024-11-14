import torch
import socket
import struct
from torchstate.C.utils import copy_bytes_to_tensor

class StateClient:
    def __init__(self, url: str):
        self.hostname, port = url.split(":")
        self.port = int(port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def get_tensor(self, path: str, inplace_tensor: torch.Tensor):
        if len(path) > 252:
            raise ValueError("Path length must be at most 252 characters")

        encoded_path = path.encode().ljust(252, b'\x00')
        dtype = 4
        size = 10
        self.client_socket.connect((self.hostname, self.port))
        packed_data = struct.pack('252sii', encoded_path, dtype, size)
        print(len(encoded_path), len(packed_data))

        # Send packed data to the server
        self.client_socket.sendall(packed_data)

        # Receive and print server response
        response = self.client_socket.recv(dtype * size)
        print(f"Received from server {len(response)}: {response}")
        copy_bytes_to_tensor(inplace_tensor, response)

        # Close the client socket
        self.client_socket.close()
