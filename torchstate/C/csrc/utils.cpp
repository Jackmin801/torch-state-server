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

// Define the Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_bytes_to_tensor", &copy_bytes_to_tensor, 
          "Copy bytes into tensor storage");
}
