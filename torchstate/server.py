import torch
from typing import Any, Dict
import struct
import socket
import threading
from torchstate.C.utils import get_bytes_from_tensor
from torchstate.logging import get_logger

class StateServerError(Exception):
    pass

def get_nested_value(d: Dict, path: str) -> Any:
    """Extract nested value from dictionary using path notation.
    
    Raises:
        StateServerError: If the path is not found in the dictionary.
    """
    try:
        parts = path.strip('[]').split('][')
        current = d
        for part in parts:
            if part.isdigit():
                part = int(part)
            current = current[part]
        return current
    except (KeyError, TypeError):
        raise StateServerError(f"Path {path} not found in state dictionary")

class StateServer:
    def __init__(self, state_dict: dict, host: str = "0.0.0.0", port: int = 12345):
        self.state_dict = state_dict
        self.host = host
        self.port = port
        self._listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._running = False
        self._server_thread = None
        self._logger = get_logger("StateServer")

    def start(self):
        """Start the server in a separate thread."""
        if self._server_thread is not None:
            raise RuntimeError("Server is already running")
        
        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop)
        self._server_thread.daemon = True  # Thread will exit when main program exits
        self._server_thread.start()
        self._logger.info(f"Server started on {self.host}:{self.port}")

    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual client connections."""
        try:
            data = client_socket.recv(252 + 4 + 4)
            if data:
                path, dtype, size = struct.unpack('252sii', data)
                path = path.decode().strip('\x00')  # Decode and strip padding
                self._logger.info("%s:%d %s %d %d", client_address[0], client_address[1], path, dtype, size)

                value = get_nested_value(self.state_dict, path)
                if dtype == 0:
                    if isinstance(value, float):
                        data = struct.pack('d', value)
                    elif isinstance(value, int):
                        data = struct.pack('q', value)
                    elif isinstance(value, str):
                        data = struct.pack('256s', value)
                    else:
                        raise ValueError("Value is not float, int or str")
                else:
                    assert isinstance(value, torch.Tensor)
                    data = get_bytes_from_tensor(value)
                client_socket.sendall(data)
        except Exception as e:
            self._logger.error(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()

    def _server_loop(self):
        """Main server loop running in separate thread."""
        self._listen_socket.bind((self.host, self.port))
        self._listen_socket.listen()
        
        while self._running:
            try:
                client_socket, client_address = self._listen_socket.accept()
                # Create a new thread for each client connection
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                if self._running:  # Only print error if we're still meant to be running
                    self._logger.error(f"Error accepting connection: {e}")

    def stop(self):
        """Stop the server gracefully."""
        if self._server_thread is None:
            return

        self._running = False
        # Create a dummy connection to unblock accept()
        try:
            dummy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            dummy_socket.connect((self.host, self.port))
            dummy_socket.close()
        except Exception:
            pass
        
        self._server_thread.join()
        self._server_thread = None
        self.close()
        self._logger.info("Server stopped")

    def close(self):
        """Close the listening socket."""
        try:
            self._listen_socket.close()
        except Exception as e:
            self._logger.error(f"Error closing socket: {e}")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass