import struct
import socket
from torchstate.C.utils import get_bytes_from_tensor

class StateServer:
    def __init__(self, state_dict: dict, host: str = "0.0.0.0", port: int = 12345):
        self.state_dict = state_dict
        self.host = host
        self.port = port
        self._listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        self._listen_socket.bind((self.host, self.port))
        self._listen_socket.listen()

        while True:
            # Accept a connection
            client_socket, client_address = self._listen_socket.accept()
            print(f"Connected by {client_address}")

            # Handle the connection
            data = client_socket.recv(252 + 4 + 4)
            print(len(data))
            if data:
                path, dtype, size = struct.unpack('252sii', data)
                path = path.decode().strip('\x00')  # Decode and strip padding

                print(f"Received path: {path}, dtype: {dtype}, size: {size}")
                data = get_bytes_from_tensor(self.state_dict['tensor'])
                print(f"Sending data {len(data)}: {data}")
                client_socket.sendall(data)

            # Close the connection
            client_socket.close()
    
    def close(self):
        self._listen_socket.close()
    
    def __del__(self):
        try:
            self.close()
        except Exception as e:
            pass
