# Graph Generation Guide

This document explains all the scripts that generate graphs in this project, what they demonstrate, and which implementations they include.

## Overview

The project has two types of demonstration scripts:
1. **Problem Demonstration** - Shows the three problems with Basic LBF
2. **Solution Demonstration** - Shows how Enhanced LBF solves those problems

---

## 1. Problem Demonstration Scripts

### `experiments/problem_demonstration.py`

**Purpose**: Demonstrates the THREE CRITICAL PROBLEMS with Basic Learned Bloom Filters.

**Implementations Tested**:
- ✅ Standard Bloom Filter (baseline)
- ✅ Basic Learned Bloom Filter (showing problems)
- ❌ Enhanced LBF NOT included (intentional - this shows the problems BEFORE the solution)

**Generated Graphs**:

| Graph | File | What It Shows | Better Direction |
|-------|------|---------------|------------------|
| Cache Problem | `cache_problem.png` | Query time distribution: Basic LBF is 297x slower | Lower is Better |
| Retraining Problem | `retraining_problem.png` | O(n) complexity: Update time scales linearly | Lower is Better |
| FPR Instability | `fpr_instability.png` | FPR variance ±800% across workloads | Lower & Stable is Better |

**Key Finding**: Basic LBF has severe performance issues that need to be solved.

**Usage**:
```bash
python experiments/problem_demonstration.py
```

---

## 2. Solution Demonstration Scripts

### `experiments/solutions_demonstration.py`

**Purpose**: Demonstrates how Enhanced LBF SOLVES the three problems with side-by-side comparisons.

**Implementations Tested**:
- ✅ Standard Bloom Filter (baseline)
- ✅ Basic Learned Bloom Filter (showing the problem)
- ✅ Cache-Aligned LBF (Solution 1)
- ✅ Incremental LBF (Solution 2)
- ✅ Adaptive LBF (Solution 3)

**Generated Graphs**:

| Graph | File | What It Shows | Better Direction |
|-------|------|---------------|------------------|
| Cache Solution | `cache_solution.png` | Basic LBF vs Cache-Aligned: 3.4x speedup | Higher Throughput is Better |
| Incremental Solution | `incremental_solution.png` | O(n) vs O(1): 133,595x faster updates | Lower Time & Higher Speedup is Better |
| Adaptive Solution | `adaptive_solution.png` | Unstable vs Stable: ±800% → ±10% variance | Lower & Stable is Better |

**Key Finding**: Enhanced LBF successfully solves all three problems.

**Usage**:
```bash
python experiments/solutions_demonstration.py
```

---

## 3. Comprehensive Testing Scripts

### `tests/test_real_world_performance.py`

**Purpose**: Tests ALL implementations on real-world datasets with comprehensive metrics.

**Implementations Tested**:
- ✅ Standard BF
- ✅ Basic LBF
- ✅ Adaptive LBF
- ✅ Cache-Aligned LBF
- ✅ Incremental LBF
- ✅ **Combined LBF (Enhanced)** ✅

**Generated Graphs**:

| Graph | File | Subplots | Better Direction |
|-------|------|----------|------------------|
| Real-World Performance | `real_world_performance.png` | 4 subplots: FPR, Query Time, Memory, Training Time | Lower is Better (all metrics) |

**Key Finding**: Combined Enhanced LBF achieves the best balance across all metrics.

**Usage**:
```bash
python tests/test_real_world_performance.py
```

---

## 4. Comparative Analysis Scripts

### `benchmarks/comparative_analysis_realworld.py`

**Purpose**: Comprehensive comparison of 7 different Bloom Filter variants on real-world datasets.

**Implementations Tested**:
- ✅ Standard Bloom Filter
- ✅ Counting Bloom Filter
- ✅ Scalable Bloom Filter
- ✅ Cuckoo Filter
- ✅ Vacuum Filter
- ✅ Basic Learned BF
- ✅ **Enhanced Learned BF** ✅

**Datasets Used**:
- URL Blacklist (malicious vs benign URLs)
- Network Traces (DDoS vs normal traffic)
- Genomic K-mers (DNA sequences)
- Database Keys (cache simulation)

**Output**: JSON results file + console comparison tables

**Key Finding**: Enhanced LBF achieves 0.1-0.3% FPR (5x better than Standard BF).

**Usage**:
```bash
python benchmarks/comparative_analysis_realworld.py
```

**Results Saved To**: `data/results/realworld_comparative_analysis.json`

---

## 5. Comprehensive Benchmark Suite

### `benchmarks/comprehensive_benchmark.py`

**Purpose**: Full performance evaluation across 6 metrics for all implementations.

**Implementations Tested**:
- ✅ Standard BF
- ✅ Basic LBF
- ✅ Cache-Aligned LBF
- ✅ Incremental LBF
- ✅ Adaptive LBF
- ✅ **Combined Enhanced LBF** ✅

**Benchmarks**:
1. Query Throughput (queries/sec)
2. Update Latency (ms)
3. FPR Stability (variance %)
4. Memory Usage (MB)
5. Cache Performance (hit rate %)
6. Scalability (performance at different sizes)

**Key Finding**: Combined Enhanced LBF achieves 1.64x throughput improvement with ±10% FPR stability.

**Usage**:
```bash
python benchmarks/comprehensive_benchmark.py
```

**Results Saved To**: `data/results/benchmark_results_<timestamp>.json`

---

## 6. Validation Scripts

These scripts validate specific enhancements and generate focused graphs:

### `validation/validate_cache_alignment.py`
- **Graph**: `validation/cache_throughput_comparison.png`
- **Shows**: Throughput (ops/sec) [Higher is Better]
- **Compares**: Basic LBF vs Cache-Aligned LBF

### `validation/validate_incremental_learning.py`
- **Graph**: `validation/incremental_complexity.png`
- **Shows**: Update Time (ms) [Lower is Better] & Speedup [Higher is Better]
- **Compares**: O(n) retrain vs O(1) incremental

### `validation/validate_adaptive_control.py`
- **Graphs**: 
  - `validation/fpr_stability.png` - FPR over time [Lower & Stable is Better]
  - `validation/pid_comparison.png` - PID controller response [Lower & Stable is Better]
- **Compares**: Different PID configurations

### `validation/validate_pattern_learning.py`
- **Graphs**:
  - `validation/pattern_learning_progression.png` - Accuracy across phases [Higher is Better]
  - `validation/continuous_learning.png` - Accuracy over epochs [Higher is Better]
- **Compares**: Enhanced LBF vs Cuckoo vs Standard BF

**Usage**:
```bash
python validation/validate_cache_alignment.py
python validation/validate_incremental_learning.py
python validation/validate_adaptive_control.py
python validation/validate_pattern_learning.py
```

---

## Summary: Which Graphs Include Enhanced LBF?

| Script | Enhanced LBF Included? | Reason |
|--------|------------------------|--------|
| `problem_demonstration.py` | ❌ NO | Shows problems BEFORE solution |
| `solutions_demonstration.py` | ✅ YES | Shows how Enhanced LBF solves problems |
| `test_real_world_performance.py` | ✅ YES | Comprehensive testing |
| `comparative_analysis_realworld.py` | ✅ YES | Full comparison with 7 variants |
| `comprehensive_benchmark.py` | ✅ YES | Performance evaluation |
| Validation scripts | ✅ YES | Each tests specific enhancement |

---

## Graph Labels Guide

All graphs now include helpful labels to guide readers:

- **[Lower is Better]** - Query time, memory usage, FPR, training time
- **[Higher is Better]** - Throughput, speedup, accuracy, cache hit rate
- **[Lower & Stable is Better]** - FPR variance, stability metrics

---

## Quick Start: Generate All Graphs

To regenerate all comparison graphs:

```bash
# 1. Problem demonstration (no Enhanced LBF - shows problems)
python experiments/problem_demonstration.py

# 2. Solution demonstration (with Enhanced LBF - shows solutions)
python experiments/solutions_demonstration.py

# 3. Comprehensive testing
python tests/test_real_world_performance.py

# 4. Comparative analysis
python benchmarks/comparative_analysis_realworld.py

# 5. Validation graphs
python validation/validate_cache_alignment.py
python validation/validate_incremental_learning.py
python validation/validate_adaptive_control.py
python validation/validate_pattern_learning.py
```

All graphs will be saved to:
- Main results: `data/results/*.png`
- Validation: `validation/*.png`

---

## Understanding the Story

The graphs tell a complete story:

1. **Problem Demonstration** (`problem_demonstration.py`)
   - Shows: Basic LBF has 3 critical problems
   - Evidence: 297x slower, O(n) updates, ±800% FPR variance

2. **Solution Demonstration** (`solutions_demonstration.py`)
   - Shows: Enhanced LBF solves all 3 problems
   - Evidence: 3.4x faster, O(1) updates, ±10% FPR variance

3. **Real-World Validation** (other scripts)
   - Shows: Enhanced LBF performs well on actual datasets
   - Evidence: 0.1-0.3% FPR, 240K+ ops/sec, 24-29x faster than Basic LBF

This narrative structure makes the research contribution clear and compelling.

---

## Need Help?

- All graph-generating scripts support `--help` (if implemented)
- Check `AGENTS.md` for detailed testing and benchmarking instructions
- See `README.md` for project overview and performance claims
- Review `docs/METHODOLOGY.md` for testing methodology details
