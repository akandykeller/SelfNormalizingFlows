#include <torch/extension.h>

#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

at::Tensor backward_weight(
    c10::ArrayRef<long int> weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

  return at::cudnn_convolution_backward_weight(
      weight_size,
      grad_output,
      input,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      true);
}


at::Tensor backward_input(
    c10::ArrayRef<long int> input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic)  {

  return at::cudnn_convolution_backward_input(
    input_size,
    grad_output,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    true);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward_weight", &backward_weight, "Conv2d backward cudnn");
  m.def("backward_input", &backward_input, "Conv2d backward cudnn");
}