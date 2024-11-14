#include <torch/extension.h>
#include <vector>

// Function to copy bytes into a tensor
void copy_bytes_to_tensor(torch::Tensor tensor, const std::string& bytes) {
    // Ensure the tensor is contiguous
    tensor = tensor.contiguous();

    // Get the number of elements in the tensor
    int64_t num_elements = tensor.numel();

    // Ensure the byte array has enough elements
    TORCH_CHECK(bytes.size() == num_elements * tensor.element_size(), 
                "Byte array size must match tensor storage size.");

    // Get a pointer to the tensor's raw data
    void* tensor_data = tensor.data_ptr();

    // Copy the bytes into the tensor's storage
    std::memcpy(tensor_data, bytes.data(), bytes.size());
}

// Function to get bytes from tensor
py::bytes get_bytes_from_tensor(torch::Tensor tensor) {
    // Ensure the tensor is contiguous
    tensor = tensor.contiguous();
    // Get the number of elements in the tensor
    int64_t num_elements = tensor.numel();
    // Calculate total size in bytes
    size_t total_bytes = num_elements * tensor.element_size();
    // Get a pointer to the tensor's raw data
    const char* tensor_data = static_cast<const char*>(tensor.data_ptr());
    // Create and return a Python bytes object
    return py::bytes(tensor_data, total_bytes);
}

// Define the Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_bytes_to_tensor", &copy_bytes_to_tensor, 
          "Copy bytes into tensor storage");
    m.def("get_bytes_from_tensor", &get_bytes_from_tensor,
          "Get bytes from tensor storage");
}