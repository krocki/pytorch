#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <limits>
#include <tuple>
#include <vector>
#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linear_cross_entropy_native.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/max.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/log.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/where.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/div.h>
#endif

namespace at::native {

enum class ChunkingStrategy {
    NAIVE,
    VOCAB_CHUNKING,
    BATCH_CHUNKING
};

// Shared thresholds for chunking decisions (kept in sync with CPU implementation)
constexpr int64_t kVocabChunkSize = 4096;
constexpr int64_t kBatchChunkSize = 1024;

Tensor naive_linear_cross_entropy_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing);

// Forward declarations
Tensor batch_chunking_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing);

// Strategy selection helper mirroring the CPU heuristics so "auto" behaves
// consistently across devices.
inline ChunkingStrategy select_chunking_strategy_cuda(
    int64_t vocab_size,
    int64_t total_batch_size,
    c10::string_view strategy) {
  if (strategy == "none") {
    return ChunkingStrategy::NAIVE;
  }
  if (strategy == "vocab") {
    return ChunkingStrategy::VOCAB_CHUNKING;
  }
  if (strategy == "batch") {
    return ChunkingStrategy::BATCH_CHUNKING;
  }
  TORCH_CHECK(strategy == "auto",
      "Unknown chunking strategy: ", strategy,
      ". Valid options: 'auto', 'vocab', 'batch', 'none'");

  const bool vocab_large = vocab_size > kVocabChunkSize;
  const bool batch_large = total_batch_size > kBatchChunkSize;

  if (!vocab_large && !batch_large) {
    return ChunkingStrategy::NAIVE;
  }
  if (vocab_large && !batch_large) {
    return ChunkingStrategy::VOCAB_CHUNKING;
  }
  if (!vocab_large && batch_large) {
    return ChunkingStrategy::BATCH_CHUNKING;
  }

  const double vocab_reduction = 1.0 - static_cast<double>(kVocabChunkSize) / static_cast<double>(vocab_size);
  const double batch_reduction = 1.0 - static_cast<double>(kBatchChunkSize) / static_cast<double>(total_batch_size);
  return (vocab_reduction >= batch_reduction) ? ChunkingStrategy::VOCAB_CHUNKING
                                              : ChunkingStrategy::BATCH_CHUNKING;
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
  
  // Flatten inputs for batched GEMM and per-sample reductions
  const auto input_flat = input.view({-1, input.size(-1)});  // [N, H]
  const auto target_flat = target.view({-1});                // [N]
  const auto valid_mask = at::ne(target_flat, ignore_index);

  const int64_t vocab_size = weight.size(0);
  const int64_t chunk_size = kVocabChunkSize;

  // When the vocabulary easily fits into a single chunk there is no benefit to
  // running the streaming algorithm. Fall back to the naive path to avoid the
  // extra kernel launches that triggered the small-model slowdown alert in the
  // CUDA milestone tests.
  if (vocab_size <= chunk_size) {
    return naive_linear_cross_entropy_cuda(
        input,
        weight,
        target,
        bias_opt,
        reduction,
        ignore_index,
        label_smoothing);
  }

  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

  const auto options = input_flat.options();
  auto long_options = options.dtype(at::kLong);
  const double neg_inf = -std::numeric_limits<double>::infinity();

  Tensor running_max = at::full({input_flat.size(0)}, neg_inf, options);
  Tensor exp_sums = at::zeros({input_flat.size(0)}, options);
  Tensor target_logits = at::zeros({input_flat.size(0)}, options);
  Tensor target_found = at::zeros({input_flat.size(0)}, long_options);
  Tensor sum_logits;
  if (label_smoothing > 0.0) {
    sum_logits = at::zeros({input_flat.size(0)}, options);
  }

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);

    auto weight_chunk = weight.slice(0, start_idx, end_idx);

    std::optional<Tensor> bias_chunk;
    if (bias_opt.has_value()) {
      bias_chunk = bias_opt->slice(0, start_idx, end_idx);
    }

    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);  // [N, chunk]

    if (label_smoothing > 0.0) {
      sum_logits = at::add(sum_logits, at::sum(logits_chunk, {-1}));
    }

    auto chunk_max = std::get<0>(logits_chunk.max(-1));
    auto new_max = at::maximum(running_max, chunk_max);

    auto exp_scale_old = at::exp(at::sub(running_max, new_max));
    auto shifted_logits = at::sub(logits_chunk, new_max.unsqueeze(-1));
    auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
    exp_sums = at::add(at::mul(exp_sums, exp_scale_old), exp_chunk);
    running_max = new_max;

    auto lower_bound = target_flat.ge(start_idx);
    auto upper_bound = target_flat.lt(end_idx);
    auto target_chunk_mask = at::logical_and(valid_mask, lower_bound);
    target_chunk_mask = at::logical_and(target_chunk_mask, upper_bound);

    auto indices = target_chunk_mask.nonzero().view({-1});
    if (indices.numel() > 0) {
      auto selected_targets = at::index_select(target_flat, 0, indices);
      auto local_targets = selected_targets.add(-start_idx);
      auto selected_logits = at::index_select(logits_chunk, 0, indices);
      auto gathered = at::gather(selected_logits, 1, local_targets.unsqueeze(1)).squeeze(1);
      target_logits.index_put_({indices}, gathered);
      auto ones_update = at::ones(indices.sizes(), long_options);
      target_found.index_put_({indices}, ones_update);
    }
  }

  auto target_found_mask = target_found.gt(0);
  auto coverage_mask = at::logical_or(target_found_mask, at::logical_not(valid_mask));
  TORCH_CHECK(coverage_mask.all().item<bool>(),
      "linear_cross_entropy_cuda: target index not found in vocabulary chunks");

  auto logsumexp = running_max.add(exp_sums.log());
  Tensor losses;
  if (label_smoothing > 0.0) {
    const double smoothing = label_smoothing;
    const double uniform = smoothing / static_cast<double>(vocab_size);
    auto main_term = target_logits.mul(1.0 - smoothing);
    auto uniform_term = sum_logits.mul(uniform);
    losses = logsumexp.sub(main_term);
    losses = losses.sub(uniform_term);
  } else {
    losses = logsumexp.sub(target_logits);
  }

  auto invalid_mask = at::logical_not(valid_mask);
  losses.masked_fill_(invalid_mask, 0);

  if (reduction == Reduction::None) {
    return losses.view(target.sizes());
  }

  auto total_loss = losses.sum();
  if (reduction == Reduction::Sum) {
    return total_loss;
  }

  const int64_t valid_count = valid_mask.sum().item<int64_t>();
  if (valid_count == 0) {
    return total_loss;
  }
  return total_loss.div(valid_count);
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
  auto logits = at::linear(input, weight, bias_opt);
  
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
  const int64_t batch_outer = input.size(0);
  const int64_t seq_len = input.dim() == 3 ? input.size(1) : 1;
  const int64_t total_batch = batch_outer * seq_len;

  ChunkingStrategy resolved = select_chunking_strategy_cuda(vocab_size, total_batch, chunking_strategy);
  switch (resolved) {
    case ChunkingStrategy::VOCAB_CHUNKING:
      return vocab_chunking_cuda(input, weight, target, bias_opt, reduction, ignore_index, label_smoothing);
    case ChunkingStrategy::BATCH_CHUNKING:
      return batch_chunking_cuda(input, weight, target, bias_opt, reduction, ignore_index, label_smoothing);
    case ChunkingStrategy::NAIVE:
    default:
      return naive_linear_cross_entropy_cuda(input, weight, target, bias_opt, reduction, ignore_index, label_smoothing);
  }
}

// CUDA batch chunking implementation for Phase 4c
// Mirrors CPU batch_chunking_cpu() algorithm using cuBLAS/cuDNN operations
// Inspired by Liger Kernel approach but maintains PyTorch library reuse strategy
Tensor batch_chunking_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  
  // Input validation - ensure all tensors are on CUDA device
  TORCH_CHECK(input.is_cuda(), "batch_chunking_cuda: input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "batch_chunking_cuda: weight must be CUDA tensor");
  TORCH_CHECK(target.is_cuda(), "batch_chunking_cuda: target must be CUDA tensor");
  TORCH_CHECK(input.device() == weight.device(), 
              "batch_chunking_cuda: input and weight must be on same device");
  TORCH_CHECK(input.device() == target.device(), 
              "batch_chunking_cuda: input and target must be on same device");
  
  // Validate bias if provided
  if (bias_opt.has_value()) {
    const auto& bias = bias_opt.value();
    TORCH_CHECK(bias.is_cuda(), "batch_chunking_cuda: bias must be CUDA tensor");
    TORCH_CHECK(bias.device() == input.device(), 
                "batch_chunking_cuda: bias must be on same device as input");
  }
  
  // Flatten multi-dimensional inputs for processing (mirrors CPU implementation)
  // This allows handling both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
  const auto input_flat = input.view({-1, input.size(-1)});  // [N, H] where N = batch * seq_len
  const auto target_flat = target.view({-1});                // [N] flattened targets
  
  const int64_t batch_size = input_flat.size(0);
  const int64_t chunk_size = kBatchChunkSize;  // Same optimal chunk size as CPU implementation (empirically validated)
  
  // Early exit if batch is too small for chunking (mirrors CPU logic)
  // Use naive implementation to avoid chunking overhead
  if (batch_size <= chunk_size) {
    auto logits = at::linear(input_flat, weight, bias_opt);
    return at::cross_entropy_loss(logits, target_flat, /*weight=*/std::nullopt, 
                                 reduction, ignore_index, label_smoothing);
  }
  
  const int64_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;
  
  Tensor losses_buffer;
  if (reduction == Reduction::None) {
    losses_buffer = at::zeros({batch_size}, input.options());
  }

  auto total_loss = at::zeros({}, input.options());
  auto valid_total = at::zeros({}, target_flat.options().dtype(at::kLong));
  
  // Process input in batch chunks to avoid materializing large logit tensors
  // Each chunk computes: [chunk_size, hidden] @ [hidden, vocab] -> [chunk_size, vocab]
  // This is the key operation: smaller batch × full vocab instead of full batch × full vocab
  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, batch_size);
    
    // Skip empty chunks (edge case safety, mirrors CPU implementation)
    if (start_idx >= end_idx) continue;
    
    // Extract batch chunk - memory efficient slicing using CUDA tensors
    // These operations are optimized by PyTorch for contiguous memory access
    auto input_chunk = input_flat.slice(0, start_idx, end_idx);   // [actual_chunk_size, H]
    auto target_chunk = target_flat.slice(0, start_idx, end_idx); // [actual_chunk_size]
    
    // Use cuBLAS-optimized matrix multiplication (at::linear uses cuBLAS internally)
    // This leverages NVIDIA's highly optimized GEMM kernels
    // Key insight: smaller batch × full vocab instead of full batch × full vocab
    auto logits_chunk = at::linear(input_chunk, weight, bias_opt); // [actual_chunk_size, vocab_size]

    auto valid_mask = at::ne(target_chunk, ignore_index);
    valid_total = at::add(valid_total, valid_mask.sum());

    const auto ce_reduction = (reduction == Reduction::None) ? Reduction::None : Reduction::Sum;
    auto chunk_loss = at::cross_entropy_loss(
        logits_chunk,
        target_chunk,
        /*weight=*/std::nullopt,
        ce_reduction,
        ignore_index,
        label_smoothing);

    if (reduction == Reduction::None) {
      auto destination = losses_buffer.slice(0, start_idx, end_idx);
      destination.copy_(chunk_loss);
      destination.masked_fill_(at::logical_not(valid_mask), 0);
    } else {
      total_loss = at::add(total_loss, chunk_loss);
    }
  }
  
  if (reduction == Reduction::None) {
    return losses_buffer.view(target.sizes());
  }

  const int64_t valid_count = valid_total.item<int64_t>();
  if (reduction == Reduction::Sum || valid_count == 0) {
    return total_loss;
  }

  return at::div(total_loss, valid_count);
}

// CUDA backward reuses the same chunking logic as the CPU path but keeps every
// intermediate on device so we stay within the memory budget established by the
// forward kernels.  All heavy lifting is delegated to existing ATen operators,
// which in turn dispatch to cuBLAS/cuDNN.
namespace {

inline Tensor cast_grad_output_cuda(const Tensor& grad_output, const Tensor& input) {
  return grad_output.to(input.scalar_type());
}

inline Tensor mask_invalid_rows_cuda(const Tensor& tensor, const Tensor& valid_mask) {
  auto mask = valid_mask.to(tensor.scalar_type()).unsqueeze(1);
  return at::mul(tensor, mask);
}

inline Tensor zeros_like_tensor_cuda(const Tensor& src) {
  return at::_ops::zeros_like::call(src, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

inline Tensor zeros_like_or_undef_cuda(const std::optional<Tensor>& opt) {
  if (opt.has_value()) {
    return zeros_like_tensor_cuda(opt.value());
  }
  return Tensor();
}

inline void apply_target_updates_cuda(
    Tensor& grad_chunk,
    const Tensor& target_flat,
    const Tensor& rows,
    int64_t offset,
    double label_smoothing) {
  if (rows.numel() == 0) {
    return;
  }
  auto selected_targets = at::index_select(target_flat, 0, rows);
  auto local_targets = selected_targets.add(-offset).to(at::kLong);
  auto gather = grad_chunk.index({rows, local_targets}).add(-(1.0 - label_smoothing));
  grad_chunk.index_put_({rows, local_targets}, gather);
}

inline void scale_grad_chunk_cuda(
    Tensor& grad_chunk,
    const Tensor& grad_output_tensor,
    const Tensor& grad_output_flat,
    int64_t reduction,
    int64_t valid_count) {
  if (reduction == Reduction::None) {
    grad_chunk.mul_(grad_output_flat.unsqueeze(1));
    return;
  }
  if (reduction == Reduction::Sum) {
    grad_chunk.mul_(grad_output_tensor);
    return;
  }
  if (valid_count == 0) {
    grad_chunk.zero_();
    return;
  }
  auto scale = grad_output_tensor.div(static_cast<double>(valid_count));
  grad_chunk.mul_(scale);
}

// GPU vocabulary chunking backward: rebuild logsumexp using the same streaming
// pass as forward, then revisit each vocabulary slice to accumulate gradients.
inline std::tuple<Tensor, Tensor, std::optional<Tensor>> backward_vocabulary_chunking_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  const auto input_flat = input.view({-1, input.size(-1)});
  const auto target_flat = target.view({-1});
  const auto dtype = input.scalar_type();
  const auto options = input.options();

  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor_cuda(input);
    Tensor grad_weight = zeros_like_tensor_cuda(weight);
    Tensor grad_bias = zeros_like_or_undef_cuda(bias_opt);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
  }

  const int64_t vocab_size = weight.size(0);
  const int64_t chunk_size = kVocabChunkSize;
  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

  Tensor running_max = at::full({input_flat.size(0)}, -std::numeric_limits<double>::infinity(), options).to(dtype);
  Tensor exp_sums = at::zeros({input_flat.size(0)}, options).to(dtype);

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
    auto weight_chunk = weight.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (bias_opt.has_value()) {
      bias_chunk = bias_opt->slice(0, start_idx, end_idx);
    }
    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);
    auto chunk_max = std::get<0>(logits_chunk.max(-1));
    auto new_max = at::maximum(running_max, chunk_max);
    auto exp_scale_old = at::exp(running_max.sub(new_max));
    auto shifted_logits = logits_chunk.sub(new_max.unsqueeze(-1));
    auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
    exp_sums = at::add(at::mul(exp_sums, exp_scale_old), exp_chunk);
    running_max = new_max;
  }

  Tensor logsumexp = running_max.add(exp_sums.log());
  Tensor grad_input = zeros_like_tensor_cuda(input_flat);
  Tensor grad_weight = zeros_like_tensor_cuda(weight);
  Tensor grad_bias = zeros_like_or_undef_cuda(bias_opt);

  const double uniform_component = label_smoothing > 0.0 ? label_smoothing / static_cast<double>(vocab_size) : 0.0;
  Tensor grad_output_tensor = cast_grad_output_cuda(grad_output, input);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
    auto weight_chunk = weight.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (bias_opt.has_value()) {
      bias_chunk = bias_opt->slice(0, start_idx, end_idx);
    }
    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);
    auto grad_chunk = at::exp(logits_chunk.sub(logsumexp.unsqueeze(-1)));
    if (label_smoothing > 0.0) {
      grad_chunk = grad_chunk.add(-uniform_component);
    }
    grad_chunk = mask_invalid_rows_cuda(grad_chunk, valid_mask);
    auto lower_bound = target_flat.ge(start_idx);
    auto upper_bound = target_flat.lt(end_idx);
    auto target_chunk_mask = at::logical_and(valid_mask, lower_bound);
    target_chunk_mask = at::logical_and(target_chunk_mask, upper_bound);
    auto rows = target_chunk_mask.nonzero().squeeze(-1);
    apply_target_updates_cuda(grad_chunk, target_flat, rows, start_idx, label_smoothing);
    scale_grad_chunk_cuda(grad_chunk, grad_output_tensor, grad_output_flat, reduction, valid_count);
    grad_chunk = mask_invalid_rows_cuda(grad_chunk, valid_mask);
    grad_input.add_(grad_chunk.matmul(weight_chunk));
    grad_weight.slice(0, start_idx, end_idx).add_(grad_chunk.transpose(0, 1).matmul(input_flat));
    if (grad_bias.defined()) {
      grad_bias.slice(0, start_idx, end_idx).add_(grad_chunk.sum(0));
    }
  }

  grad_input = grad_input.view_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
}

// GPU batch chunking backward mirrors the CPU implementation but keeps the
// working set on device by visiting at most `chunk_size` rows at a time.
inline std::tuple<Tensor, Tensor, std::optional<Tensor>> backward_batch_chunking_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size) {
  const auto input_flat = input.view({-1, input.size(-1)});
  const auto target_flat = target.view({-1});
  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor_cuda(input);
    Tensor grad_weight = zeros_like_tensor_cuda(weight);
    Tensor grad_bias = zeros_like_or_undef_cuda(bias_opt);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
  }

  Tensor grad_input = zeros_like_tensor_cuda(input_flat);
  Tensor grad_weight = zeros_like_tensor_cuda(weight);
  Tensor grad_bias = zeros_like_or_undef_cuda(bias_opt);

  Tensor grad_output_tensor = cast_grad_output_cuda(grad_output, input);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  const double uniform_component = label_smoothing > 0.0 ? label_smoothing / static_cast<double>(weight.size(0)) : 0.0;
  const int64_t total = input_flat.size(0);

  for (int64_t start_idx = 0; start_idx < total; start_idx += chunk_size) {
    const int64_t slice = std::min<int64_t>(chunk_size, total - start_idx);
    auto input_chunk = input_flat.narrow(0, start_idx, slice);
    auto target_chunk = target_flat.narrow(0, start_idx, slice);
    auto valid_mask_chunk = valid_mask.narrow(0, start_idx, slice);
    auto logits_chunk = at::linear(input_chunk, weight, bias_opt);
    auto logsumexp_chunk = at::_ops::logsumexp::call(logits_chunk, std::vector<int64_t>{1}, false);
    auto grad_chunk = at::exp(logits_chunk.sub(logsumexp_chunk.unsqueeze(-1)));
    if (label_smoothing > 0.0) {
      grad_chunk = grad_chunk.add(-uniform_component);
    }
    grad_chunk = mask_invalid_rows_cuda(grad_chunk, valid_mask_chunk);
    auto rows = valid_mask_chunk.nonzero().squeeze(-1);
    if (rows.numel() > 0) {
      auto targets_slice = at::index_select(target_chunk, 0, rows).to(at::kLong);
      auto gather = grad_chunk.index({rows, targets_slice}).add(-(1.0 - label_smoothing));
      grad_chunk.index_put_({rows, targets_slice}, gather);
    }
    if (reduction == Reduction::None) {
      grad_chunk.mul_(grad_output_flat.narrow(0, start_idx, slice).unsqueeze(1));
    } else if (reduction == Reduction::Sum) {
      grad_chunk.mul_(grad_output_tensor);
    } else {
      if (valid_count == 0) {
        continue;
      }
      grad_chunk.mul_(grad_output_tensor.div(static_cast<double>(valid_count)));
    }
    grad_chunk = mask_invalid_rows_cuda(grad_chunk, valid_mask_chunk);
    grad_input.narrow(0, start_idx, slice).add_(grad_chunk.matmul(weight));
    grad_weight.add_(grad_chunk.transpose(0, 1).matmul(input_chunk));
    if (grad_bias.defined()) {
      grad_bias.add_(grad_chunk.sum(0));
    }
  }

  grad_input = grad_input.view_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
}

} // anonymous namespace

std::tuple<Tensor, Tensor, std::optional<Tensor>> linear_cross_entropy_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy) {
  TORCH_CHECK(input.dim() >= 2, "linear_cross_entropy_backward_cuda: input must have at least 2 dimensions");
  TORCH_CHECK(weight.dim() == 2, "linear_cross_entropy_backward_cuda: weight must be 2-dimensional");
  TORCH_CHECK(input.size(-1) == weight.size(1),
      "linear_cross_entropy_backward_cuda: input.size(-1) must match weight.size(1)");

  const int64_t vocab_size = weight.size(0);
  const int64_t batch_outer = input.size(0);
  const int64_t seq_len = input.dim() == 3 ? input.size(1) : 1;
  const int64_t total_batch = batch_outer * seq_len;

  ChunkingStrategy resolved_strategy = select_chunking_strategy_cuda(vocab_size, total_batch, chunking_strategy);
  if (resolved_strategy == ChunkingStrategy::VOCAB_CHUNKING && vocab_size <= kVocabChunkSize) {
    resolved_strategy = ChunkingStrategy::NAIVE;
  }

  if (resolved_strategy == ChunkingStrategy::VOCAB_CHUNKING) {
    return backward_vocabulary_chunking_cuda(
        input,
        weight,
        target,
        bias_opt,
        grad_output,
        reduction,
        ignore_index,
        label_smoothing);
  }

  const int64_t default_chunk = resolved_strategy == ChunkingStrategy::BATCH_CHUNKING
      ? kBatchChunkSize
      : input.view({-1, input.size(-1)}).size(0);
  return backward_batch_chunking_cuda(
      input,
      weight,
      target,
      bias_opt,
      grad_output,
      reduction,
      ignore_index,
      label_smoothing,
      default_chunk);
}

} // namespace at::native
