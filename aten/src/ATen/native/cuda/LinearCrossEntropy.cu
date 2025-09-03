#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linear_cross_entropy_native.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/where.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/div.h>
#endif

namespace at::native {

// Utility function for strategy selection heuristics
// Based on empirical analysis from memory profiling (Milestone 2)
inline bool should_use_vocab_chunking_cuda(int64_t vocab_size, int64_t batch_size) {
  // Use vocabulary chunking for large vocabularies (LLM training scenarios)
  // Threshold based on proven CPU implementation and memory constraints
  return vocab_size > 8192;
}

// Apply final reduction based on reduction mode
// Handles mean/sum reduction consistently with PyTorch cross_entropy behavior
Tensor apply_reduction_cuda(const Tensor& total_loss, int64_t valid_count, int64_t reduction) {
  if (reduction == Reduction::Mean) {
    if (valid_count > 0) {
      return at::div(total_loss, valid_count);
    } else {
      return total_loss; // Will be 0 if no valid samples
    }
  } else if (reduction == Reduction::Sum) {
    return total_loss;
  } else { // Reduction::None
    TORCH_CHECK(false, "Reduction::None not supported for vocabulary chunking yet");
  }
}

// CUDA vocabulary chunking implementation
// Based on established approaches from PyTorch Issue #124480 and proven CPU implementation
// Uses cuBLAS for matrix operations and cuDNN for cross-entropy (library reuse strategy)
Tensor vocab_chunking_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  
  // Input validation - ensure all tensors are on CUDA device
  TORCH_CHECK(input.is_cuda(), "linear_cross_entropy_cuda: input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "linear_cross_entropy_cuda: weight must be CUDA tensor");
  TORCH_CHECK(target.is_cuda(), "linear_cross_entropy_cuda: target must be CUDA tensor");
  TORCH_CHECK(input.device() == weight.device(), 
              "linear_cross_entropy_cuda: input and weight must be on same device");
  TORCH_CHECK(input.device() == target.device(), 
              "linear_cross_entropy_cuda: input and target must be on same device");
  
  // Validate bias if provided
  if (bias_opt.has_value()) {
    const auto& bias = bias_opt.value();
    TORCH_CHECK(bias.is_cuda(), "linear_cross_entropy_cuda: bias must be CUDA tensor");
    TORCH_CHECK(bias.device() == input.device(), 
                "linear_cross_entropy_cuda: bias must be on same device as input");
  }
  
  // Flatten multi-dimensional inputs (mirrors proven CPU implementation)
  // Handles both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
  const auto input_flat = input.view({-1, input.size(-1)});  // [N, H] where N = batch * seq_len
  const auto target_flat = target.view({-1});                // [N] flattened targets
  
  const int64_t vocab_size = weight.size(0);
  const int64_t chunk_size = 4096;  // Same optimal chunk size as CPU implementation
  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;
  
  // Initialize accumulators on GPU
  auto total_loss = at::zeros({}, input.options());
  int64_t valid_count = 0;
  
  // Process each vocabulary chunk using native CUDA operations
  // This mirrors the CPU algorithm but uses cuBLAS/cuDNN instead of CPU operations
  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
    
    // Extract vocabulary chunk (key insight from Issue #124480)
    // Process weight matrix in vocabulary dimension to avoid large logit tensor
    auto weight_chunk = weight.slice(0, start_idx, end_idx);  // [chunk_vocab_size, hidden_dim]
    
    // Extract corresponding bias chunk if bias is provided
    std::optional<Tensor> bias_chunk;
    if (bias_opt.has_value()) {
      bias_chunk = bias_opt->slice(0, start_idx, end_idx);
    }
    
    // Use cuBLAS-optimized matrix multiplication (at::linear uses cuBLAS internally)
    // Only materializes [N, chunk_size] instead of [N, vocab_size] tensor - key memory savings
    auto logits_chunk = at::linear(input_flat, weight_chunk.t(), bias_chunk);  // [N, chunk_vocab_size]
    
    // Create boolean mask for samples with targets in current vocabulary chunk
    // This implements selective loss computation from vocabulary chunking theory
    auto target_mask = at::logical_and(
        at::ge(target_flat, start_idx),   // target >= start_idx
        at::lt(target_flat, end_idx)      // target < end_idx
    );
    
    // Only process chunks that contain relevant targets (optimization)
    if (target_mask.any().item().toBool()) {
      // Adjust target indices to chunk-local indices (0-based within chunk)
      auto target_chunk = at::sub(target_flat, start_idx);
      
      // Apply ignore_index mask (convert out-of-range targets to ignore_index)
      target_chunk = at::where(target_mask, target_chunk, ignore_index);
      
      // Use cuDNN-optimized cross-entropy computation
      // This leverages the same optimized kernels used by standard PyTorch cross_entropy
      auto chunk_loss = at::cross_entropy_loss(
          logits_chunk,           // [N, chunk_vocab_size] logits
          target_chunk,           // [N] chunk-local target indices
          /*weight=*/std::nullopt, // no class weights (can be added in future)
          Reduction::Sum,         // sum within chunk, apply final reduction later
          ignore_index,           // ignore specified index
          label_smoothing         // label smoothing factor
      );
      
      // Accumulate results across chunks (GPU-native operations)
      total_loss += chunk_loss;
      valid_count += target_mask.sum().item().toLong();
    }
  }
  
  // Apply final reduction (mean/sum) based on accumulated results
  return apply_reduction_cuda(total_loss, valid_count, reduction);
}

// Naive CUDA implementation for small vocabularies
// Uses standard PyTorch operations without chunking (no memory optimization needed)
Tensor naive_linear_cross_entropy_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  
  // Standard linear + cross_entropy operations for small models
  // This path avoids chunking overhead when memory savings aren't needed
  auto logits = at::linear(input, weight.t(), bias_opt);
  
  // Reshape tensors for cross_entropy compatibility
  // cross_entropy expects [N, C] logits and [N] targets
  auto logits_flat = logits.view({-1, logits.size(-1)});  // [N, C]
  auto target_flat = target.view({-1});                   // [N]
  
  return at::cross_entropy_loss(
      logits_flat, target_flat,
      /*weight=*/std::nullopt,
      reduction,
      ignore_index,
      label_smoothing
  );
}

// Main CUDA implementation entry point
// Implements Phase 4a: CUDA Vocabulary Chunking (replaces CPU delegation)
Tensor linear_cross_entropy_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy) {
  
  // Input dimension validation (matches CPU implementation requirements)
  TORCH_CHECK(input.dim() >= 2, "linear_cross_entropy_cuda: input must have at least 2 dimensions");
  TORCH_CHECK(weight.dim() == 2, "linear_cross_entropy_cuda: weight must be 2-dimensional");
  TORCH_CHECK(input.size(-1) == weight.size(1), 
              "linear_cross_entropy_cuda: input.size(-1) must match weight.size(1)");
  
  const int64_t vocab_size = weight.size(0);
  const int64_t batch_size = input.view({-1, input.size(-1)}).size(0);
  
  // Strategy selection based on input characteristics
  if (chunking_strategy == "vocab" || 
      (chunking_strategy == "auto" && should_use_vocab_chunking_cuda(vocab_size, batch_size))) {
    // Use vocabulary chunking for large vocabularies (LLM training scenarios)
    return vocab_chunking_cuda(input, weight, target, bias_opt, reduction, ignore_index, label_smoothing);
  } else if (chunking_strategy == "batch") {
    // Placeholder for Phase 4c implementation
    TORCH_CHECK(false, "Batch chunking not yet implemented for CUDA. Use 'vocab', 'auto', or 'none'.");
  } else {
    // Use naive implementation for small models (no chunking overhead)
    return naive_linear_cross_entropy_cuda(input, weight, target, bias_opt, reduction, ignore_index, label_smoothing);
  }
}

} // namespace at::native