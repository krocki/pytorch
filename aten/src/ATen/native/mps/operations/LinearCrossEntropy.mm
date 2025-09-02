#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linear_cross_entropy_native.h>
#endif

namespace at::native {

Tensor linear_cross_entropy_mps(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy) {
  
  // For now, delegate to CPU implementation
  // TODO: Implement optimized MPS kernels for vocabulary and batch chunking
  return at::native::linear_cross_entropy_cpu(
      input.to(kCPU), 
      weight.to(kCPU), 
      target.to(kCPU),
      bias_opt.has_value() ? std::optional<Tensor>(bias_opt->to(kCPU)) : std::nullopt,
      reduction, 
      ignore_index, 
      label_smoothing, 
      chunking_strategy
  ).to(input.device());
}

} // namespace at::native