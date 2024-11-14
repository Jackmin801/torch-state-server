import struct
import socket
import threading
from torchstate.C.utils import get_bytes_from_tensor

class StateServer:
    def __init__(self, state_dict: dict, host: str = "0.0.0.0", port: int = 12345):
        self.state_dict = state_dict
        self.host = host
        self.port = port
        self._listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._running = False
        self._server_thread = None

    def start(self):
        """Start the server in a separate thread."""
        if self._server_thread is not None:
            raise RuntimeError("Server is already running")
        
        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop)
        self._server_thread.daemon = True  # Thread will exit when main program exits
        self._server_thread.start()
        print(f"Server started on {self.host}:{self.port}")

    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual client connections."""
        try:
            print(f"Connected by {client_address}")
            data = client_socket.recv(252 + 4 + 4)
            print(len(data))
            if data:
                path, dtype, size = struct.unpack('252sii', data)
                path = path.decode().strip('\x00')  # Decode and strip padding
                print(f"Received path: {path}, dtype: {dtype}, size: {size}")
                data = get_bytes_from_tensor(self.state_dict['tensor'])
                print(f"Sending data {len(data)}: {data}")
                client_socket.sendall(data)
        except Exception as e:
            print(f"Error handling client {client_address}: {e}")
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
                    print(f"Error accepting connection: {e}")

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
        print("Server stopped")

    def close(self):
        """Close the listening socket."""
        try:
            self._listen_socket.close()
        except Exception as e:
            print(f"Error closing socket: {e}")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass