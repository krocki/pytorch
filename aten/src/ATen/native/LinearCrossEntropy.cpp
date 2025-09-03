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
#include <ATen/ops/ne.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/linear_cross_entropy_native.h>
#endif

namespace at::native {

// Strategy selection for optimal chunking approach
enum class ChunkingStrategy {
    NAIVE,           // No chunking - standard approach
    VOCAB_CHUNKING,  // Chunk vocabulary dimension (existing)
    BATCH_CHUNKING   // Chunk batch dimension (new)
};

// Determine optimal chunking strategy based on input dimensions and user preference
// Based on memory reduction analysis and empirical validation
inline ChunkingStrategy select_chunking_strategy(
    int64_t vocab_size, 
    int64_t batch_size, 
    int64_t seq_len, 
    c10::string_view strategy) {
    
    if (strategy == "none") {
        return ChunkingStrategy::NAIVE;
    } else if (strategy == "vocab") {
        return ChunkingStrategy::VOCAB_CHUNKING;
    } else if (strategy == "batch") {
        return ChunkingStrategy::BATCH_CHUNKING;
    } else if (strategy == "auto") {
        // Empirically validated chunk sizes for optimal memory/compute balance
        const int64_t vocab_chunk_size = 4096;   // Same as existing implementation
        const int64_t batch_chunk_size = 1024;   // Optimized for batch processing
        
        const int64_t total_batch_size = batch_size * seq_len;
        
        // Determine which dimensions benefit from chunking
        bool vocab_large = vocab_size > vocab_chunk_size;
        bool batch_large = total_batch_size > batch_chunk_size;
        
        if (!vocab_large && !batch_large) {
            return ChunkingStrategy::NAIVE;
        } else if (vocab_large && !batch_large) {
            return ChunkingStrategy::VOCAB_CHUNKING;
        } else if (!vocab_large && batch_large) {
            return ChunkingStrategy::BATCH_CHUNKING;
        } else {
            // Both dimensions are large - choose strategy with better memory reduction
            // Memory reduction = 1 - (chunk_size / total_size)
            double vocab_reduction = 1.0 - static_cast<double>(vocab_chunk_size) / vocab_size;
            double batch_reduction = 1.0 - static_cast<double>(batch_chunk_size) / total_batch_size;
            
            return (vocab_reduction >= batch_reduction) ? 
                   ChunkingStrategy::VOCAB_CHUNKING : ChunkingStrategy::BATCH_CHUNKING;
        }
    } else {
        TORCH_CHECK(false, "Unknown chunking strategy: ", strategy, 
                   ". Valid options: 'auto', 'vocab', 'batch', 'none'");
    }
}

// Apply final reduction based on reduction mode
// Handles mean/sum reduction consistently across all chunking strategies
inline Tensor apply_final_reduction(const Tensor& total_loss, int64_t valid_count, int64_t reduction) {
    if (reduction == Reduction::Mean) {
        if (valid_count > 0) {
            return at::div(total_loss, valid_count);
        } else {
            return total_loss; // Will be 0 if no valid samples
        }
    } else if (reduction == Reduction::Sum) {
        return total_loss;
    } else { // Reduction::None
        TORCH_CHECK(false, "Reduction::None not supported for chunking strategies yet");
    }
}

// Batch chunking implementation for CPU
// Inspired by Liger Kernel approach: processes input in batch chunks to reduce memory usage
// Memory reduction: [N, V] -> [chunk_size, V] where N = batch_size * seq_len
Tensor batch_chunking_cpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
    
    // Flatten multi-dimensional inputs for processing (standard PyTorch pattern)
    // This allows handling both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
    auto input_flat = input.view({-1, input.size(-1)});  // [N, H] where N = batch * seq_len
    auto target_flat = target.view({-1});                // [N] flattened targets
    
    const int64_t batch_size = input_flat.size(0);
    const int64_t chunk_size = 1024;  // Empirically optimized for batch dimension chunking
    
    // Get bias tensor if provided
    const Tensor& bias = bias_opt.value_or(Tensor());
    
    // Early exit if batch is too small for chunking
    if (batch_size <= chunk_size) {
        auto logits = at::linear(input_flat, weight.t(), bias);
        return at::cross_entropy_loss(logits, target_flat, std::nullopt, reduction, ignore_index, label_smoothing);
    }
    
    const int64_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;
    
    Tensor total_loss = at::zeros({}, input.options());
    int64_t valid_count = 0;
    
    // Process input in batch chunks to avoid materializing large logit tensors
    // Each chunk computes: [chunk_size, hidden] @ [hidden, vocab] -> [chunk_size, vocab]
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int64_t start_idx = chunk_idx * chunk_size;
        int64_t end_idx = std::min(start_idx + chunk_size, batch_size);
        
        // Skip empty chunks (edge case safety)
        if (start_idx >= end_idx) continue;
        
        // Extract batch chunk - memory efficient slicing
        auto input_chunk = input_flat.slice(0, start_idx, end_idx);   // [actual_chunk_size, H]
        auto target_chunk = target_flat.slice(0, start_idx, end_idx); // [actual_chunk_size]
        
        // Compute logits for this batch chunk (full vocabulary)
        // This is the key operation: smaller batch × full vocab instead of full batch × full vocab
        auto logits_chunk = at::linear(input_chunk, weight.t(), bias); // [actual_chunk_size, vocab_size]
        
        // Compute cross entropy for this chunk
        auto chunk_loss = at::cross_entropy_loss(logits_chunk, target_chunk,
                                                std::nullopt,    // no class weights for now
                                                Reduction::Sum,  // sum within chunk, apply final reduction later
                                                ignore_index,
                                                label_smoothing);
        
        // Accumulate results across chunks
        total_loss += chunk_loss;
        
        // Count valid samples (excluding ignore_index) for mean reduction
        auto valid_mask = at::ne(target_chunk, ignore_index);
        valid_count += valid_mask.sum().item().toLong();
    }
    
    // Apply final reduction (mean/sum) based on accumulated results
    return apply_final_reduction(total_loss, valid_count, reduction);
}

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
  
  // CPU linear_cross_entropy implementation with dual chunking strategy support
  // 
  // CHUNKING STRATEGIES:
  // 1. Vocabulary Chunking: Based on PyTorch Issue #124480
  //    - Chunks weight matrix in vocabulary dimension
  //    - Optimal for: Large vocab (>50k), moderate batch → LLM training
  //    - Memory reduction: [N, V] -> [N, chunk_size] where V = vocab_size
  // 
  // 2. Batch Chunking: Inspired by Liger Kernel approach
  //    - Chunks input tensor in batch dimension  
  //    - Optimal for: Large batch (>2k), moderate vocab → Fine-tuning
  //    - Memory reduction: [N, V] -> [chunk_size, V] where N = batch * seq_len
  //
  // The core insight is to avoid materializing massive logit tensors.
  // Example: batch=8, seq_len=4096, vocab=256k creates a 16.8GB logit tensor.
  // Chunking reduces this to manageable sizes based on the bottleneck dimension.
  
  // Calculate input dimensions for strategy selection
  const int64_t vocab_size = weight.size(0);
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.dim() == 3 ? input.size(1) : 1;
  
  // Select optimal chunking strategy based on input characteristics and user preference
  ChunkingStrategy selected_strategy = select_chunking_strategy(vocab_size, batch_size, seq_len, chunking_strategy);
  
  // Execute selected chunking strategy
  if (selected_strategy == ChunkingStrategy::VOCAB_CHUNKING) {
    // Vocabulary chunking implementation (based on PyTorch Issue #124480 approach)
    const int64_t chunk_size = 4096;  // Empirically validated chunk size for optimal memory/compute balance
    
    const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;
    
    // Flatten multi-dimensional inputs for processing (standard PyTorch pattern)
    // This allows handling both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
    auto input_flat = input.view({-1, input.size(-1)});  // [N, H] where N = batch * seq_len
    auto target_flat = target.view({-1});                // [N] flattened targets
    
    Tensor total_loss = at::zeros({}, input.options());
    int64_t valid_count = 0;
    
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      int64_t start_idx = chunk_idx * chunk_size;
      int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
      
      // Extract vocabulary chunk (key insight from Issue #124480)
      // Process weight matrix in vocabulary dimension to avoid large logit tensor
      auto weight_chunk = weight.slice(0, start_idx, end_idx);  // [chunk_vocab_size, hidden_dim]
      
      // Extract corresponding bias chunk if bias is provided
      std::optional<Tensor> bias_chunk;
      if (bias.defined()) {
        bias_chunk = bias.slice(0, start_idx, end_idx);
      }
      
      // Compute partial logits using standard linear operation (memory efficient)
      // Only materializes [N, chunk_size] instead of [N, vocab_size] tensor
      auto logits_chunk = at::linear(input_flat, weight_chunk.t(), bias_chunk);  // [N, chunk_vocab_size]
      
      // Create boolean mask for samples with targets in current vocabulary chunk
      // This implements the selective loss computation from vocabulary chunking theory
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
    return apply_final_reduction(total_loss, valid_count, reduction);
    
  } else if (selected_strategy == ChunkingStrategy::BATCH_CHUNKING) {
    // Batch chunking implementation - call dedicated function
    return batch_chunking_cpu(input, weight, target, bias_opt, reduction, ignore_index, label_smoothing);
    
  } else { // ChunkingStrategy::NAIVE
    // Naive implementation for small models or when chunking not beneficial
    auto logits = at::linear(input, weight.t(), bias);
    
    // Reshape tensors for cross_entropy: logits [N, C] and target [N]
    auto logits_flat = logits.view({-1, logits.size(-1)});  // [N, C]
    auto target_flat = target.view({-1});                   // [N]
    
    return at::cross_entropy_loss(logits_flat, target_flat, std::nullopt, reduction, ignore_index, label_smoothing);
  }
}

} // namespace at::native