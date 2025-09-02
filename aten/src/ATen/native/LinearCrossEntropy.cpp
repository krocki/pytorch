#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/where.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/div.h>
#include <ATen/ops/linear_cross_entropy_native.h>
#endif

namespace at::native {

Tensor linear_cross_entropy_cpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy) {
  
  // Validate inputs
  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, got ", input.dim());
  TORCH_CHECK(weight.dim() == 2, "Expected weight to be 2-dimensional, got ", weight.dim());
  TORCH_CHECK(input.size(-1) == weight.size(1), 
              "Expected input.size(-1) to match weight.size(1), got ", 
              input.size(-1), " and ", weight.size(1));
  
  // Get bias tensor if provided
  const Tensor& bias = bias_opt.value_or(Tensor());
  
  // For CPU implementation, we start with the naive approach
  // TODO: Implement vocabulary chunking for CPU
  
  if (chunking_strategy == "vocab" || chunking_strategy == "auto") {
    // Vocabulary chunking implementation
    const int64_t vocab_size = weight.size(0);
    const int64_t chunk_size = 4096;  // Configurable chunk size
    
    if (vocab_size > chunk_size) {
      // Implement chunked computation
      const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;
      
      // Flatten input to [batch_size * seq_len, hidden_dim] for easier processing  
      auto input_flat = input.view({-1, input.size(-1)});  // [N, H]
      auto target_flat = target.view({-1});                // [N]
      
      Tensor total_loss = at::zeros({}, input.options());
      int64_t valid_count = 0;
      
      for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int64_t start_idx = chunk_idx * chunk_size;
        int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
        
        // Extract weight chunk: [chunk_vocab_size, hidden_dim]
        auto weight_chunk = weight.slice(0, start_idx, end_idx);
        
        // Extract bias chunk if provided
        std::optional<Tensor> bias_chunk;
        if (bias.defined()) {
          bias_chunk = bias.slice(0, start_idx, end_idx);
        }
        
        // Compute logits for this chunk: [N, chunk_vocab_size]
        auto logits_chunk = at::linear(input_flat, weight_chunk.t(), bias_chunk);
        
        // Create mask for targets in this chunk's vocabulary range  
        auto target_mask = at::logical_and(at::ge(target_flat, start_idx), at::lt(target_flat, end_idx));
        
        if (target_mask.any().item().toBool()) {
          // Adjust target indices to chunk-local indices
          auto target_chunk = at::sub(target_flat, start_idx);
          
          // Apply ignore_index mask (convert out-of-range to ignore_index)
          target_chunk = at::where(target_mask, target_chunk, ignore_index);
          
          // Compute cross entropy for this chunk
          auto chunk_loss = at::cross_entropy_loss(logits_chunk, target_chunk, 
                                                  std::nullopt, // no class weights for now
                                                  Reduction::Sum, // sum within chunk
                                                  ignore_index, 
                                                  label_smoothing);
          
          total_loss += chunk_loss;
          valid_count += target_mask.sum().item().toLong();
        }
      }
      
      // Apply final reduction
      if (reduction == Reduction::Mean) {
        if (valid_count > 0) {
          return at::div(total_loss, valid_count);
        } else {
          return total_loss; // Will be 0
        }
      } else if (reduction == Reduction::Sum) {
        return total_loss;
      } else { // None
        TORCH_CHECK(false, "Reduction::None not supported for vocabulary chunking yet");
      }
    }
  }
  
  // Fallback to naive implementation for small vocabularies or batch chunking
  auto logits = at::linear(input, weight.t(), bias);
  
  // Reshape tensors for cross_entropy: logits [N, C] and target [N]
  auto logits_flat = logits.view({-1, logits.size(-1)});  // [N, C]
  auto target_flat = target.view({-1});                   // [N]
  
  return at::cross_entropy_loss(logits_flat, target_flat, std::nullopt, reduction, ignore_index, label_smoothing);
}

} // namespace at::native