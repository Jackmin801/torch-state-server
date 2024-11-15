import torch
from typing import Any, Dict, Union, Optional
import struct
import socket
import threading
from torchstate.C.utils import get_bytes_from_tensor
from torchstate.logging import get_logger
from torchstate.ttype_consts import TransferType, ScalarTransferType

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

    def _pack_error_response(self, error_msg: str) -> bytes:
        """Pack an error response to send back to the client."""
        encoded_msg = error_msg.encode()
        return struct.pack('iiq', 1, ScalarTransferType.STR.value, len(encoded_msg)) + encoded_msg

    def _pack_tensor_metadata(self, tensor: torch.Tensor, transfer_type: int) -> bytes:
        """Pack tensor metadata including shape and stride information."""
        shape = list(tensor.shape) + [-1] * (6 - len(tensor.shape))
        stride = list(tensor.stride()) + [-1] * (6 - len(tensor.stride()))
        return struct.pack('iiqiiiiiiiiiiii',
                         0,  # success
                         transfer_type,
                         tensor.numel(),
                         *shape,
                         *stride)

    def _pack_scalar_response(self, value: Union[float, int, bool, str], scalar_type: ScalarTransferType) -> bytes:
        """Pack a scalar response with appropriate format."""
        if scalar_type == ScalarTransferType.STR:
            encoded_str = value.encode()
            return struct.pack('iiq', 0, scalar_type.value, len(encoded_str)) + encoded_str
        elif scalar_type == ScalarTransferType.FLOAT64:
            return struct.pack('iiqd', 0, scalar_type.value, 8, float(value))
        elif scalar_type == ScalarTransferType.INT64:
            return struct.pack('iiqq', 0, scalar_type.value, 8, int(value))
        elif scalar_type == ScalarTransferType.BOOL8:
            return struct.pack('iiq?', 0, scalar_type.value, 1, bool(value))
        else:
            raise ValueError(f"Unsupported scalar type: {scalar_type}")

    def _get_transfer_type(self, tensor: torch.Tensor, requested_type: Optional[int]) -> int:
        """Determine the appropriate transfer type for a tensor."""
        if requested_type != -1:
            return requested_type
        
        if tensor.dtype == torch.float32:
            return TransferType.FLOAT32.value
        elif tensor.dtype == torch.bfloat16:
            return TransferType.BFLOAT16.value
        elif tensor.dtype == torch.float16:
            return TransferType.FLOAT16.value
        else:
            raise StateServerError(f"Unsupported tensor type: {tensor.dtype}")

    def _handle_tensor_request(
        self,
        client_socket: socket.socket,
        value: torch.Tensor, 
        transfer_type: int,
        size: int
    ) -> None:
        """Handle a tensor request and send the appropriate response."""
        actual_type = self._get_transfer_type(value, transfer_type)
        
        if size != -1 and size != value.numel():
            raise StateServerError(f"Requested size {size} doesn't match tensor size {value.numel()}")

        # Send metadata or simple header based on whether size was specified
        if size == -1:
            client_socket.sendall(self._pack_tensor_metadata(value, actual_type))
        else:
            client_socket.sendall(struct.pack('iiq', 0, actual_type, value.numel()))

        # Send codebook if needed (for quantized types)
        if actual_type == TransferType.UNIFORM_INT8.value:
            # TODO: Implement codebook generation and sending
            codebook = bytes([0] * 256)  # Placeholder
            client_socket.sendall(codebook)

        # Send tensor data
        tensor_bytes = get_bytes_from_tensor(value)
        client_socket.sendall(tensor_bytes)

    def _handle_scalar_request(self, client_socket: socket.socket, value: Any, 
                             scalar_type: ScalarTransferType) -> None:
        """Handle a scalar request and send the appropriate response."""
        try:
            response = self._pack_scalar_response(value, scalar_type)
            client_socket.sendall(response)
        except (ValueError, struct.error) as e:
            error_response = self._pack_error_response(f"Error packing scalar value: {str(e)}")
            client_socket.sendall(error_response)

    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual client connections."""
        try:
            # Receive request header (244 bytes path + 4 bytes type + 8 bytes size)
            data = client_socket.recv(256)
            if not data or len(data) != 256:
                raise StateServerError("Invalid request format")

            # Unpack request
            path, transfer_type, size = struct.unpack('244siq', data)
            path = path.decode().strip('\x00')
            
            self._logger.info("%s:%d %s %d %d", client_address[0], client_address[1], 
                            path, transfer_type, size)

            # Get value from state dictionary
            value = get_nested_value(self.state_dict, path)

            # Handle tensor requests
            if transfer_type == -1 or transfer_type >= TransferType.FLOAT32.value:
                if not isinstance(value, torch.Tensor):
                    raise StateServerError(f"Value at path {path} is not a tensor")
                self._handle_tensor_request(client_socket, value, transfer_type, size)
            
            # Handle scalar requests
            elif transfer_type in [t.value for t in ScalarTransferType]:
                scalar_type = ScalarTransferType(transfer_type)
                self._handle_scalar_request(client_socket, value, scalar_type)
            
            else:
                raise StateServerError(f"Unsupported transfer type: {transfer_type}")

        except Exception as e:
            self._logger.error(f"Error handling client {client_address}: {e}")
            try:
                error_response = self._pack_error_response(str(e))
                client_socket.sendall(error_response)
            except Exception as send_error:
                self._logger.error(f"Error sending error response: {send_error}")

        finally:
            client_socket.close()

    def start(self):
        """Start the server in a separate thread."""
        if self._server_thread is not None:
            raise RuntimeError("Server is already running")
        
        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop)
        self._server_thread.daemon = True
        self._server_thread.start()
        self._logger.info(f"Server started on {self.host}:{self.port}")

    def _server_loop(self):
        """Main server loop running in separate thread."""
        self._listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen_socket.bind((self.host, self.port))
        self._listen_socket.listen()
        
        while self._running:
            try:
                client_socket, client_address = self._listen_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                if self._running:
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