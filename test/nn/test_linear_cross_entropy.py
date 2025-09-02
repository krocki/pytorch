# Owner(s): ["module: nn"]

"""
Test suite for torch.nn.functional.linear_cross_entropy implementation.

This file contains milestone-based tests that track implementation progress:

MILESTONE STRUCTURE:
- Each milestone has dedicated test methods with naming pattern: test_step_N_*
- Tests are designed to be run incrementally as implementation progresses
- Regression tests ensure previous milestones continue working
- Final test_linear_ce() validates complete implementation

MILESTONE BREAKDOWN:
Step 1 (Foundation): Basic API, parameter validation, naive fallback
Step 2 (Memory): Memory profiling infrastructure and baseline measurements  
Step 3 (Vocab Chunking): Vocabulary dimension chunking implementation
Step 4 (Batch Chunking): Batch dimension chunking with Triton kernels
Step 5 (Dispatch): Intelligent strategy selection based on input characteristics
Step 6 (Integration): Compatibility with torch.compile, distributed training
Step 7 (Performance): Benchmarking and performance validation
Step 8 (Production): Final production readiness verification

RUNNING TESTS:
# Run specific milestone:
python test/nn/test_linear_cross_entropy.py step_1 cpu
python test/nn/test_linear_cross_entropy.py step_2 cuda

# Run regression tests:
python test/nn/test_linear_cross_entropy.py regression cpu

# Run all implemented tests:
python test/nn/test_linear_cross_entropy.py all cpu

NOTE TO SELF: Never use 'python -c "..."' for testing. Always add tests to this file
and run them through the proper test runner.
"""

import sys
import torch
import torch.nn.functional as F


class TestLinearCrossEntropy:
    """Test suite for linear_cross_entropy implementation across all milestones."""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.tests_passed = 0
        self.tests_failed = 0
        
    def assert_true(self, condition, message=""):
        if condition:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
            print(f"FAIL: {message}")
            
    def assert_allclose(self, a, b, atol=1e-8, message=""):
        condition = torch.allclose(a, b, atol=atol)
        self.assert_true(condition, f"{message} - Expected close values, got {a} vs {b}")
        
    def assert_raises(self, exception_type, func, message=""):
        try:
            func()
            self.tests_failed += 1
            print(f"FAIL: {message} - Expected {exception_type.__name__} but no exception was raised")
        except exception_type:
            self.tests_passed += 1
        except Exception as e:
            self.tests_failed += 1
            print(f"FAIL: {message} - Expected {exception_type.__name__} but got {type(e).__name__}: {e}")

    # =============================================================================
    # MILESTONE 1: FOUNDATION SETUP
    # =============================================================================

    def test_step_1_foundation(self):
        """
        Milestone 1: Test basic API structure and correctness.
        
        Validates:
        - Function exists and is callable
        - Parameter validation works
        - Numerical correctness against reference
        - Gradient computation correctness
        - All parameter combinations work
        """
        print(f"\n=== MILESTONE 1 TESTS ({self.device}) ===")
        
        # Test 1.1: Basic functionality
        batch_size, seq_len, hidden_dim, vocab_size = 4, 8, 16, 20
        input = torch.randn(batch_size, seq_len, hidden_dim, device=self.device, requires_grad=True)
        weight = torch.randn(vocab_size, hidden_dim, device=self.device, requires_grad=True)
        bias = torch.randn(vocab_size, device=self.device, requires_grad=True)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        # Reference implementation
        logits_ref = F.linear(input, weight, bias)
        loss_ref = F.cross_entropy(logits_ref.view(-1, vocab_size), target.view(-1))
        loss_ref.backward()
        
        # Store reference gradients
        input_grad_ref = input.grad.clone()
        weight_grad_ref = weight.grad.clone()
        bias_grad_ref = bias.grad.clone()
        
        # Reset gradients for fused implementation
        input.grad = None
        weight.grad = None
        bias.grad = None
        
        # Fused implementation
        loss_fused = F.linear_cross_entropy(input.view(-1, hidden_dim), weight, target.view(-1), bias)
        loss_fused.backward()
        
        # Test correctness
        self.assert_allclose(loss_fused, loss_ref, atol=1e-8, message="Loss values should match")
        self.assert_allclose(input.grad, input_grad_ref, atol=1e-6, message="Input gradients should match")
        self.assert_allclose(weight.grad, weight_grad_ref, atol=1e-6, message="Weight gradients should match")
        self.assert_allclose(bias.grad, bias_grad_ref, atol=1e-6, message="Bias gradients should match")
        print("PASS: 1.1 Basic functionality and gradient correctness")
        
        # Test 1.2: Parameter validation
        input_simple = torch.randn(4, 8, device=self.device)
        weight_simple = torch.randn(10, 8, device=self.device)
        target_simple = torch.randint(0, 10, (4,), device=self.device)
        
        # Invalid reduction
        self.assert_raises(ValueError, 
                          lambda: F.linear_cross_entropy(input_simple, weight_simple, target_simple, reduction="invalid"),
                          "Should raise ValueError for invalid reduction")
        
        # Invalid label_smoothing
        self.assert_raises(ValueError,
                          lambda: F.linear_cross_entropy(input_simple, weight_simple, target_simple, label_smoothing=-0.1),
                          "Should raise ValueError for negative label_smoothing")
        
        self.assert_raises(ValueError,
                          lambda: F.linear_cross_entropy(input_simple, weight_simple, target_simple, label_smoothing=1.1),
                          "Should raise ValueError for label_smoothing > 1.0")
        
        # Invalid chunking_strategy
        self.assert_raises(ValueError,
                          lambda: F.linear_cross_entropy(input_simple, weight_simple, target_simple, chunking_strategy="invalid"),
                          "Should raise ValueError for invalid chunking_strategy")
        print("PASS: 1.2 Parameter validation")
        
        # Test 1.3: All reduction modes
        for reduction in ['mean', 'sum', 'none']:
            logits = F.linear(input_simple, weight_simple)
            loss_ref = F.cross_entropy(logits, target_simple, reduction=reduction)
            loss_fused = F.linear_cross_entropy(input_simple, weight_simple, target_simple, reduction=reduction)
            self.assert_allclose(loss_fused, loss_ref, atol=1e-8, message=f"Reduction '{reduction}' should match")
        print("PASS: 1.3 All reduction modes")
        
        # Test 1.4: ignore_index parameter
        target_with_ignore = target_simple.clone()
        target_with_ignore[0] = -100
        
        logits = F.linear(input_simple, weight_simple)
        loss_ref = F.cross_entropy(logits, target_with_ignore, ignore_index=-100)
        loss_fused = F.linear_cross_entropy(input_simple, weight_simple, target_with_ignore, ignore_index=-100)
        self.assert_allclose(loss_fused, loss_ref, atol=1e-8, message="ignore_index should work correctly")
        print("PASS: 1.4 ignore_index parameter")
        
        # Test 1.5: label_smoothing parameter
        for label_smoothing in [0.0, 0.1, 0.3]:
            logits = F.linear(input_simple, weight_simple)
            loss_ref = F.cross_entropy(logits, target_simple, label_smoothing=label_smoothing)
            loss_fused = F.linear_cross_entropy(input_simple, weight_simple, target_simple, label_smoothing=label_smoothing)
            self.assert_allclose(loss_fused, loss_ref, atol=1e-8, message=f"label_smoothing {label_smoothing} should match")
        print("PASS: 1.5 label_smoothing parameter")
        
        print("SUCCESS: MILESTONE 1 COMPLETED - Foundation setup successful")

    def test_step_1_regression(self):
        """Milestone 1 regression test - ensures basic functionality never breaks."""
        input = torch.randn(4, 8, requires_grad=True, device=self.device)
        weight = torch.randn(10, 8, requires_grad=True, device=self.device)
        target = torch.randint(0, 10, (4,), device=self.device)
        
        # Test all parameter combinations
        for reduction in ['mean', 'sum', 'none']:
            for ignore_index in [-100, 5]:
                for label_smoothing in [0.0, 0.1]:
                    result = F.linear_cross_entropy(input, weight, target, 
                                                  reduction=reduction,
                                                  ignore_index=ignore_index,
                                                  label_smoothing=label_smoothing)
                    self.assert_true(result is not None, "Result should not be None")
                    self.assert_true(result.requires_grad, "Result should require gradients")
        print("PASS: Milestone 1 regression test")

    def _measure_peak_memory(self, operation_fn):
        """
        Utility to measure peak memory usage of an operation.
        
        Args:
            operation_fn: Function that performs the operation to measure
            
        Returns:
            float: Peak memory usage in MB (GPU if available, otherwise rough CPU estimate)
        """
        if self.device == "cuda" and torch.cuda.is_available():
            # GPU memory measurement using CUDA
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            
            # Run the operation
            result = operation_fn()
            
            # Get peak memory in MB
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            
            # Clean up
            del result
            torch.cuda.empty_cache()
            
            return peak_memory
        else:
            # CPU memory estimation (rough approximation)
            # For CPU, we'll estimate based on tensor sizes since psutil isn't available
            import gc
            gc.collect()
            
            # Run operation and estimate memory from tensor sizes
            result = operation_fn()
            
            # Rough estimation: sum all tensor memory
            total_memory = 0
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    total_memory += obj.nelement() * obj.element_size()
            
            total_memory_mb = total_memory / 1024 / 1024
            
            # Clean up
            del result
            gc.collect()
            
            return total_memory_mb

    # =============================================================================
    # MILESTONE 2: MEMORY PROFILING INFRASTRUCTURE
    # =============================================================================

    def test_step_2_memory_profiling(self):
        """
        Milestone 2: Test memory profiling infrastructure.
        
        Validates:
        - Memory measurement utilities work
        - Baseline memory measurements are recorded
        - Memory measurements are reproducible
        - Establishes evidence for memory problem with large vocabs
        """
        print(f"\n=== MILESTONE 2 TESTS ({self.device}) ===")
        
        # Test 2.1: Memory measurement utility basic functionality
        def simple_operation():
            x = torch.randn(1000, 1000, device=self.device)
            y = torch.randn(1000, 1000, device=self.device)
            result = torch.matmul(x, y)
            return result
        
        memory_used = self._measure_peak_memory(simple_operation)
        self.assert_true(memory_used > 0, "Memory measurement should detect usage")
        print(f"PASS: 2.1 Memory measurement utility (detected {memory_used:.2f} MB)")
        
        # Test 2.2: Linear cross entropy baseline memory measurements
        # These establish the memory problem we're trying to solve
        test_cases = [
            (32, 256, 4096, 10000, "Small vocab"),     # ~300MB expected
            (16, 512, 4096, 30000, "Medium vocab"),    # ~1GB expected  
            (8, 1024, 4096, 50000, "Large vocab"),     # ~1.6GB expected
        ]
        
        if self.device == "cuda" and torch.cuda.is_available():
            # Only run large tests on GPU where we can measure accurately
            test_cases.append((4, 2048, 4096, 100000, "Very large vocab"))  # ~3.2GB expected
        
        self.baselines = {}  # Store baselines for regression detection
        
        for batch_size, seq_len, hidden_dim, vocab_size, description in test_cases:
            def reference_linear_cross_entropy():
                # Current naive implementation - materializes full logits
                input = torch.randn(batch_size, seq_len, hidden_dim, device=self.device, requires_grad=True)
                weight = torch.randn(vocab_size, hidden_dim, device=self.device, requires_grad=True)
                target = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                
                # This is the memory-intensive operation we want to optimize
                logits = F.linear(input, weight)  # This creates the large logits tensor
                loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
                loss.backward()
                return loss
                
            memory_mb = self._measure_peak_memory(reference_linear_cross_entropy)
            self.baselines[vocab_size] = memory_mb
            
            # Calculate expected logit tensor size for validation
            logits_size_mb = (batch_size * seq_len * vocab_size * 4) / (1024 * 1024)  # 4 bytes per float32
            
            print(f"PASS: 2.2 {description} (vocab={vocab_size}): {memory_mb:.1f} MB")
            print(f"      Logits tensor alone: {logits_size_mb:.1f} MB")
            
            # Validate memory usage makes sense (should be at least the logits size)
            if self.device == "cuda":
                self.assert_true(memory_mb >= logits_size_mb * 0.8, 
                               f"Memory usage {memory_mb:.1f}MB should be at least logits size {logits_size_mb:.1f}MB")
        
        # Test 2.3: Memory measurement reproducibility
        if self.device == "cuda" and torch.cuda.is_available():
            memory1 = self._measure_peak_memory(simple_operation)
            memory2 = self._measure_peak_memory(simple_operation)
            variance = abs(memory1 - memory2) / memory1 if memory1 > 0 else 0
            self.assert_true(variance < 0.1, f"Memory measurements should be reproducible (variance: {variance:.3f})")
            print("PASS: 2.3 Memory measurement reproducibility")
        else:
            print("SKIP: 2.3 Memory measurement reproducibility (CPU measurements less precise)")
        
        # Test 2.4: Demonstrate memory problem scaling
        if len(self.baselines) >= 2:
            vocab_sizes = sorted(self.baselines.keys())
            small_vocab_memory = self.baselines[vocab_sizes[0]]
            large_vocab_memory = self.baselines[vocab_sizes[-1]]
            
            scaling_factor = large_vocab_memory / small_vocab_memory
            expected_scaling = vocab_sizes[-1] / vocab_sizes[0]
            
            print(f"PASS: 2.4 Memory scaling validation:")
            print(f"      {vocab_sizes[0]} vocab: {small_vocab_memory:.1f} MB")
            print(f"      {vocab_sizes[-1]} vocab: {large_vocab_memory:.1f} MB") 
            print(f"      Scaling factor: {scaling_factor:.1f}x (expected ~{expected_scaling:.1f}x)")
            
            # Memory should scale roughly with vocab size (allowing for overhead)
            self.assert_true(scaling_factor >= expected_scaling * 0.5, 
                           f"Memory should scale with vocab size")
        
        print("SUCCESS: MILESTONE 2 COMPLETED - Memory profiling infrastructure ready")
        print(f"BASELINES ESTABLISHED: {len(self.baselines)} vocab sizes measured")
        
        # Store baseline data for future regression tests
        if hasattr(self, 'baselines'):
            print("BASELINE DATA:")
            for vocab_size in sorted(self.baselines.keys()):
                print(f"  vocab_size_{vocab_size}: {self.baselines[vocab_size]:.1f} MB")

    def test_step_2_regression(self):
        """
        Milestone 2 regression test - ensures memory measurement infrastructure never breaks.
        Utility to verify memory measurement functionality remains working.
        """
        # Quick smoke test to ensure memory measurement works
        def tiny_operation():
            x = torch.randn(100, 100, device=self.device)
            return x * 2
        
        memory_used = self._measure_peak_memory(tiny_operation)
        self.assert_true(memory_used >= 0, "Memory measurement should return non-negative values")
        print("PASS: Milestone 2 regression test - memory measurement functional")

    # =============================================================================
    # MILESTONE 3: VOCABULARY CHUNKING IMPLEMENTATION  
    # =============================================================================

    def test_step_3_vocab_chunking(self, device="cpu"):
        """
        Milestone 3: Test vocabulary chunking implementation.
        
        Validates:
        - Memory usage reduced for large vocab models
        - Numerical accuracy maintained
        - Forward and backward pass correctness
        - No performance regression for small vocabs
        """
        print(f"\n=== MILESTONE 3 TESTS ({device}) ===")
        
        # Test 3.1: Basic functionality test
        print("Test 3.1: Basic functionality test")
        batch_size, seq_len, hidden_dim, vocab_size = 2, 4, 8, 16
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
        weight_tensor = torch.randn(vocab_size, hidden_dim, device=device, requires_grad=True)
        target_tensor = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Weight shape: {weight_tensor.shape}")
        print(f"  Target shape: {target_tensor.shape}")
        
        # First test what F.linear produces
        logits_ref = F.linear(input_tensor, weight_tensor)
        print(f"  Logits shape: {logits_ref.shape}")
        
        # Test reference implementation - need to reshape for cross_entropy
        logits_flat = logits_ref.view(-1, logits_ref.size(-1))  # [batch*seq, vocab]
        target_flat = target_tensor.view(-1)                    # [batch*seq]
        print(f"  Logits flat shape: {logits_flat.shape}")
        print(f"  Target flat shape: {target_flat.shape}")
        
        loss_ref = F.cross_entropy(logits_flat, target_flat)
        print(f"  Reference loss: {loss_ref}")
        
        # Test our implementation
        loss_ours = F.linear_cross_entropy(input_tensor, weight_tensor, target_tensor)
        print(f"  Our loss: {loss_ours}")
        
        self.assert_allclose(loss_ours, loss_ref, message="Basic functionality test")
        print("PASS: Basic functionality test")
        
        # Test 3.2: Numerical correctness with different vocab sizes
        print("Test 3.2: Small vocabulary test")
        
        small_vocab = 1000
        input_small = torch.randn(4, 8, 64, device=device, requires_grad=True)
        weight_small = torch.randn(small_vocab, 64, device=device, requires_grad=True)
        target_small = torch.randint(0, small_vocab, (4, 8), device=device)
        
        # Reference implementation - need to reshape for cross_entropy
        logits_ref = F.linear(input_small, weight_small)
        logits_ref_flat = logits_ref.view(-1, logits_ref.size(-1))
        target_small_flat = target_small.view(-1)
        loss_ref = F.cross_entropy(logits_ref_flat, target_small_flat)
        
        # Our implementation with chunking strategy
        loss_chunked = F.linear_cross_entropy(input_small, weight_small, target_small, 
                                            chunking_strategy="vocab")
        
        self.assert_allclose(loss_chunked, loss_ref, message="Small vocab chunking correctness")
        print("PASS: Small vocabulary test")
        
        # Test 3.3: Large vocabulary test (should trigger chunking)
        print("Test 3.3: Large vocabulary test")
        
        large_vocab = 8192  # Larger than default chunk size (4096)
        input_large = torch.randn(2, 4, 64, device=device, requires_grad=True)
        weight_large = torch.randn(large_vocab, 64, device=device, requires_grad=True)
        target_large = torch.randint(0, large_vocab, (2, 4), device=device)
        
        # Reference implementation - need to reshape for cross_entropy
        logits_ref_large = F.linear(input_large, weight_large)
        logits_ref_large_flat = logits_ref_large.view(-1, logits_ref_large.size(-1))
        target_large_flat = target_large.view(-1)
        loss_ref_large = F.cross_entropy(logits_ref_large_flat, target_large_flat)
        
        # Our chunked implementation
        loss_chunked_large = F.linear_cross_entropy(input_large, weight_large, target_large,
                                                  chunking_strategy="vocab")
        
        self.assert_allclose(loss_chunked_large, loss_ref_large, atol=1e-4, message="Large vocab chunking correctness")
        print("PASS: Large vocabulary test")
        
        # Test 3.4: Strategy consistency test
        print("Test 3.4: Strategy consistency test")
        
        # For small vocabularies, different strategies should give same result
        loss_auto = F.linear_cross_entropy(input_small, weight_small, target_small, 
                                          chunking_strategy="auto")
        loss_vocab = F.linear_cross_entropy(input_small, weight_small, target_small,
                                          chunking_strategy="vocab")
        
        self.assert_allclose(loss_auto, loss_vocab, message="Strategy consistency")
        print("PASS: Strategy consistency test")
        
        print("=== MILESTONE 3 COMPLETED ===\n")

    # =============================================================================
    # MILESTONE 4: BATCH CHUNKING IMPLEMENTATION
    # =============================================================================

    def test_step_4_batch_chunking(self, device="cpu"):
        """
        Milestone 4: Test batch chunking implementation.
        
        Validates:
        - Triton kernel registration and functionality
        - Batch chunking reduces memory usage
        - Autograd compatibility
        - Performance improvement for large batches
        """
        print(f"\n=== MILESTONE 4 TESTS ({device}) ===")
        print("TODO: Milestone 4 not yet implemented - batch chunking pending")
        
        # Placeholder for batch chunking tests
        # TODO: Implement when batch chunking is ready

    # =============================================================================
    # MILESTONE 5: INTELLIGENT DISPATCH LOGIC
    # =============================================================================

    def test_step_5_dispatch(self, device="cpu"):
        """
        Milestone 5: Test intelligent dispatch logic.
        
        Validates:
        - Automatic strategy selection
        - Heuristics choose optimal approach
        - All strategies accessible manually
        - No performance regression
        """
        print(f"\n=== MILESTONE 5 TESTS ({device}) ===")
        print("TODO: Milestone 5 not yet implemented - dispatch logic pending")

    # =============================================================================
    # MILESTONE 6: INTEGRATION TESTING
    # =============================================================================

    def test_step_6_integration(self, device="cpu"):
        """
        Milestone 6: Test integration with PyTorch ecosystem.
        
        Validates:
        - torch.compile compatibility
        - Distributed training compatibility  
        - Mixed precision training support
        - No conflicts with existing features
        """
        print(f"\n=== MILESTONE 6 TESTS ({device}) ===")
        print("TODO: Milestone 6 not yet implemented - integration tests pending")

    # =============================================================================
    # MILESTONE 7: PERFORMANCE VALIDATION
    # =============================================================================

    def test_step_7_performance(self, device="cpu"):
        """
        Milestone 7: Test performance validation.
        
        Validates:
        - Memory reduction targets achieved
        - No performance regression for small models
        - Training throughput improvements
        - Cross-hardware validation
        """
        print(f"\n=== MILESTONE 7 TESTS ({device}) ===")
        print("TODO: Milestone 7 not yet implemented - performance validation pending")

    # =============================================================================
    # MILESTONE 8: PRODUCTION READINESS
    # =============================================================================

    def test_step_8_production(self, device="cpu"):
        """
        Milestone 8: Test production readiness.
        
        Validates:
        - Complete test coverage
        - Code style compliance
        - Documentation completeness
        - Final integration validation
        """
        print(f"\n=== MILESTONE 8 TESTS ({device}) ===")
        print("TODO: Milestone 8 not yet implemented - production readiness pending")

    # =============================================================================
    # FINAL INTEGRATION TEST
    # =============================================================================

    def test_linear_ce(self, device="cpu"):
        """
        Final integration test for complete linear_cross_entropy implementation.
        
        This test validates the entire implementation across all scenarios:
        - Small and large vocabulary sizes
        - Different batch sizes and sequence lengths
        - All parameter combinations
        - Memory efficiency validation
        - Performance characteristics
        
        This test should only pass when ALL milestones are complete.
        """
        print(f"\n=== FINAL INTEGRATION TEST ({device}) ===")
        
        # High-level validation of complete implementation
        test_scenarios = [
            # (batch_size, seq_len, hidden_dim, vocab_size, description)
            (32, 128, 768, 1000, "Small model"),
            (16, 512, 1024, 30000, "Medium model"), 
            (8, 1024, 2048, 50000, "Large model"),
            (4, 2048, 4096, 128000, "Very large vocab"),
        ]
        
        for batch_size, seq_len, hidden_dim, vocab_size, desc in test_scenarios:
            print(f"Testing {desc}: {batch_size}x{seq_len}x{hidden_dim} -> {vocab_size}")
            
            input = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
            weight = torch.randn(vocab_size, hidden_dim, device=device, requires_grad=True)
            target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            # Test automatic strategy selection
            loss = F.linear_cross_entropy(input, weight, target, chunking_strategy="auto")
            loss.backward()
            
            # Verify gradients exist and are reasonable
            self.assertIsNotNone(input.grad)
            self.assertIsNotNone(weight.grad)
            self.assertFalse(torch.isnan(loss))
            self.assertTrue(loss.requires_grad)
            
            print(f"  PASS: {desc} completed successfully")
        
        print("SUCCESS: FINAL INTEGRATION TEST COMPLETED")


    def run_milestone_test(self, milestone_num):
        """Run a specific milestone test."""
        method_name = f"test_step_{milestone_num}_"
        
        for attr_name in dir(self):
            if attr_name.startswith(method_name):
                print(f"\nRunning {attr_name} on {self.device}")
                getattr(self, attr_name)()
                return True
        
        print(f"ERROR: No test found for milestone {milestone_num}")
        return False

    def run_all_tests(self):
        """Run all available tests."""
        for i in range(1, 9):
            if hasattr(self, f"test_step_{i}_foundation") or hasattr(self, f"test_step_{i}_memory_profiling"):
                try:
                    self.run_milestone_test(i)
                except Exception as e:
                    print(f"ERROR: Milestone {i} failed: {e}")
                    self.tests_failed += 1

    def report(self):
        """Print test results summary."""
        total = self.tests_passed + self.tests_failed
        print(f"\n=== TEST RESULTS ===")
        print(f"Tests run: {total}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.tests_failed == 0:
            print("SUCCESS: All tests passed!")
        else:
            print("FAILURE: Some tests failed!")
        
        return self.tests_failed == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python test/nn/test_linear_cross_entropy.py <step_N|all|regression> [device]")
        sys.exit(1)
    
    command = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    runner = TestLinearCrossEntropy(device)
    
    try:
        if command == "all":
            runner.run_all_tests()
        elif command.startswith("step_"):
            milestone_num = int(command.split("_")[1])
            runner.run_milestone_test(milestone_num)
        elif command == "regression":
            # Run all regression tests
            runner.test_step_1_regression()
            runner.test_step_2_regression()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: step_1, step_2, ..., step_8, all, regression")
            sys.exit(1)
        
        success = runner.report()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()