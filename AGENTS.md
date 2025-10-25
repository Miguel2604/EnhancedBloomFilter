# AGENTS.md

## Project Overview

This repository implements Enhanced Learned Bloom Filters (LBF) that solve three critical problems in traditional Learned Bloom Filters:
1. Poor cache locality (70% cache miss rate)
2. Expensive O(n) retraining when data changes
3. Unstable false positive rates (±800% variance)

The solution combines machine learning with probabilistic data structures, achieving 1.64x throughput improvement, O(1) updates, and ±10% FPR stability.

## Dev Environment Tips

- Use `source venv/bin/activate` to activate the virtual environment before any work
- Run `pip install -r requirements.txt` after activating venv to ensure all dependencies are installed
- Use `python -m <module>` instead of direct imports when testing individual components
- The codebase uses Python 3.10+ features - ensure your Python version is compatible
- When adding new Bloom Filter variants, follow the pattern in `src/bloom_filter/standard.py`
- URL feature extraction (`_extract_url_features`) is critical for ML model performance - don't modify without understanding the impact

## Project Structure

```
src/
├── bloom_filter/        # Traditional implementations (baseline)
├── learned_bloom_filter/ # Basic LBF with ML
└── enhanced_lbf/        # Our enhanced versions with fixes
```

Key files to understand:
- `src/enhanced_lbf/combined.py` - Production-ready implementation with all enhancements
- `src/bloom_filter/standard.py` - Baseline for comparisons
- `experiments/problem_demonstration.py` - Shows the three problems we solve

## Testing Instructions

- Run `pytest tests/` to execute all unit tests
- Run `pytest tests/test_standard_bf.py` for basic Bloom Filter tests
- Run `pytest tests/test_real_world_performance.py` for real-world dataset tests
- Use `pytest --cov=src tests/` to check code coverage
- Run `python experiments/problem_demonstration.py` to verify the three problems exist
- Run `python validation/verify_implementations.py` to validate all enhancements work correctly
- Each enhancement can be tested individually:
  - `python src/enhanced_lbf/cache_aligned.py` - Test cache optimization
  - `python src/enhanced_lbf/incremental.py` - Test O(1) updates
  - `python src/enhanced_lbf/adaptive.py` - Test FPR stability

## Benchmarking Instructions

- Run `python benchmarks/comprehensive_benchmark.py` for full performance evaluation
- Run `python benchmarks/comparative_analysis_realworld.py` for real-world dataset comparisons
- Results are saved to `data/results/` as JSON files
- Use `python analyze_results.py` to generate performance summaries
- Expected improvements over baseline:
  - Query throughput: 1.64x
  - Update complexity: O(n) → O(1)
  - FPR variance: ±800% → ±10%

## Dataset Management

- Run `python scripts/download_datasets.py` to fetch real-world datasets (URLs, network traces, genomic data)
- Use `python scripts/safe_download_datasets.py` if the main script has issues
- Datasets are stored in `data/datasets/`
- For URL filtering tests, use `python scripts/fix_url_filtering.py` if encountering issues
- Dataset types:
  - URL blacklist: Malicious and benign URLs for security testing
  - Network traces: DDoS and normal traffic patterns
  - Genomic k-mers: DNA sequence fragments for bioinformatics
  - Database keys: Synthetic keys for caching scenarios

## Code Modification Guidelines

### When modifying the ML model:
- The model uses Passive-Aggressive learning with momentum
- Training requires BOTH positive and negative examples
- Multiple epochs (default: 3) improve initial discrimination
- Learning rate decay (0.9995) prevents overfitting

### When modifying cache optimization:
- Maintain 64-byte alignment for CPU cache lines
- SIMD operations require batch sizes of 8
- Cache blocks are pre-allocated - don't resize dynamically

### When modifying adaptive control:
- PID controller parameters (Kp=2.0, Ki=0.5, Kd=0.1) are tuned for stability
- Threshold starts at 0.7 for security applications
- Monitoring window (1000 queries) affects adaptation speed

### Critical invariants to maintain:
- Primary filter must store ALL positive items (no false negatives)
- Backup filter handles ML model false negatives
- Feature extraction must return exactly 20 features
- Thread safety is NOT implemented - add locks for concurrent access

## Common Issues and Fixes

### High False Positive Rate:
- Check training data balance (need both positive and negative examples)
- Increase training epochs in `_init_model()`
- Verify threshold is appropriate (default: 0.7)

### Poor Cache Performance:
- Ensure `enable_cache_opt=True` in initialization
- Use batch operations for SIMD benefits
- Warm cache before benchmarking

### Model Not Learning:
- Check learning rate (starts at 0.1, decays to 0.01)
- Verify sliding window size (default: 10000)
- Ensure features are being extracted correctly

### Memory Issues:
- Enhanced LBF uses ~10MB for ML model
- Reduce `window_size` or `reservoir_size` if needed
- Use standard BF for memory-constrained environments

## PR Instructions

- Title format: `[Component] Description` (e.g., `[Enhanced LBF] Fix cache alignment bug`)
- Always run `pytest tests/` before committing
- Run validation scripts for modified components
- Include benchmark results if performance is affected
- Update docstrings for any API changes
- Add tests for new functionality
- Ensure Python 3.10+ compatibility

## Performance Validation

Before merging changes that affect performance:
1. Run `python benchmarks/comprehensive_benchmark.py`
2. Compare results with baseline in `data/results/`
3. Ensure no regression in key metrics:
   - Query throughput should be >300K ops/sec
   - Update time should be <0.1ms
   - FPR variance should be <20%

## Debug Tips

- Use `verbose=True` in constructors for detailed initialization logs
- Check `filter.get_stats()` for cache hit rates and model performance
- Monitor `threshold_history` and `fpr_history` for adaptive behavior
- Use `filter.model.get_weights()` to inspect model parameters
- Enable Python profiling with `python -m cProfile` for bottleneck analysis

## Dependencies to be aware of

- `mmh3`: MurmurHash3 for hash functions - critical for BF performance
- `bitarray`: Efficient bit manipulation - don't replace with lists
- `numpy`: Vectorized operations for SIMD - version 1.21.0+ required
- `scikit-learn`: ML models - ensure compatibility with passive-aggressive classifier
- `psutil`: System monitoring - optional but useful for cache analysis