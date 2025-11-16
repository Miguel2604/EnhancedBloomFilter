#!/usr/bin/env python3
"""
LBF Comparative Benchmark - Enhanced LBF vs State-of-the-Art LBF Variations

Compares:
1. Enhanced LBF (our implementation)
2. Adaptive LBF (Ada-BF) - Dai & Shrivastava 2019
3. Partitioned LBF (PLBF) - Vaidya et al. 2020
4. Stable LBF (s-SLBF) - Liu et al. 2020

Test Scenarios:
- Static Workload (train once, query many times)
- Dynamic Workload (continuous insertions)
- Cache-Sensitive Workload (high query rate)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import gc

# Import implementations
from src.enhanced_lbf.combined import CombinedEnhancedLBF
from src.reference_lbf.ada_bf import AdaptiveLearnedBloomFilter
from src.reference_lbf.plbf import PartitionedLearnedBloomFilter
from src.reference_lbf.stable_lbf import StableLearnedBloomFilter


class LBFBenchmark:
    """Benchmark suite for Learned Bloom Filter variations."""
    
    def __init__(self, output_dir: str = "data/results/lbf_comparison"):
        """Initialize benchmark."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(dict)
        
    def load_url_dataset(self, 
                         malicious_file: str = "data/datasets/url_blacklist/malicious_urls.txt",
                         benign_file: str = "data/datasets/url_blacklist/benign_urls.txt",
                         n_samples: int = 10000) -> Tuple[List[str], List[str], List[str]]:
        """Load URL dataset for testing using prepared real-world files.

        This benchmark no longer falls back to synthetic data. If the
        expected files are missing, it will raise an error and instruct the
        user to run the dataset downloader.

        Returns:
            (train_positive, train_negative, test_positive, test_negative)
        """
        print(f"\n📂 Loading URL dataset...")

        if not os.path.exists(malicious_file) or not os.path.exists(benign_file):
            print(f"❌ Real-world URL dataset not found.")
            print(f"   Expected malicious file: {malicious_file}")
            print(f"   Expected benign file:    {benign_file}")
            print("   Please run: python scripts/download_datasets.py")
            raise FileNotFoundError("Real-world URL dataset files are missing")

        # Load malicious URLs
        with open(malicious_file, 'r') as f:
            malicious_urls = [line.strip() for line in f if line.strip()][:n_samples * 2]

        # Load benign URLs
        with open(benign_file, 'r') as f:
            benign_urls = [line.strip() for line in f if line.strip()][:n_samples * 2]

        # Split into train/test
        train_positive = malicious_urls[:n_samples]
        test_positive = malicious_urls[n_samples:n_samples * 2]
        train_negative = benign_urls[:n_samples]
        test_negative = benign_urls[n_samples:n_samples * 2]
        
        print(f"✅ Loaded {len(train_positive)} training positives")
        print(f"✅ Loaded {len(train_negative)} training negatives")
        print(f"✅ Loaded {len(test_positive)} test positives")
        print(f"✅ Loaded {len(test_negative)} test negatives")
        
        return train_positive, train_negative, test_positive, test_negative
    
    def measure_construction_time(self, filter_obj: Any, 
                                   positive_set: List[Any],
                                   negative_set: List[Any]) -> float:
        """Measure filter construction time."""
        # Already constructed, return 0 (construction happens in __init__)
        return 0.0
    
    def measure_query_performance(self, filter_obj: Any,
                                   test_items: List[Any],
                                   n_iterations: int = 10) -> Dict[str, float]:
        """
        Measure query performance.
        
        Returns:
            {
                'throughput': queries per second,
                'latency_mean': average latency in ms,
                'latency_p50': median latency in ms,
                'latency_p95': 95th percentile latency in ms,
                'latency_p99': 99th percentile latency in ms
            }
        """
        print(f"  📊 Testing query performance ({len(test_items)} items, {n_iterations} iterations)...")
        
        latencies = []
        
        # Warmup
        for item in test_items[:100]:
            filter_obj.query(item)
        
        # Actual measurement
        for _ in range(n_iterations):
            for item in test_items:
                start = time.perf_counter()
                filter_obj.query(item)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        total_queries = len(test_items) * n_iterations
        total_time = np.sum(latencies) / 1000  # Convert back to seconds
        
        return {
            'throughput': total_queries / total_time,
            'latency_mean': np.mean(latencies),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'total_queries': total_queries
        }
    
    def measure_update_performance(self, filter_obj: Any,
                                    new_items: List[Any]) -> Dict[str, float]:
        """
        Measure update/insertion performance.
        
        Returns:
            {
                'throughput': insertions per second,
                'latency_mean': average latency in ms,
                'total_insertions': number of items inserted
            }
        """
        print(f"  📊 Testing update performance ({len(new_items)} insertions)...")
        
        latencies = []
        
        for item in new_items:
            start = time.perf_counter()
            filter_obj.add(item)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        total_time = np.sum(latencies) / 1000  # Convert to seconds
        
        return {
            'throughput': len(new_items) / total_time,
            'latency_mean': np.mean(latencies),
            'total_insertions': len(new_items)
        }
    
    def measure_false_positive_rate(self, filter_obj: Any,
                                     negative_samples: List[Any]) -> float:
        """Measure false positive rate on negative samples."""
        print(f"  📊 Testing FPR on {len(negative_samples)} negative samples...")
        
        false_positives = sum(1 for item in negative_samples if filter_obj.query(item))
        fpr = false_positives / len(negative_samples)
        
        return fpr
    
    def measure_false_negative_rate(self, filter_obj: Any,
                                     positive_samples: List[Any]) -> float:
        """
        Measure false negative rate on positive samples.
        
        Note: Should be 0% for known training positives (no false negatives allowed).
        May be higher for unseen test positives (expected - LBF hasn't learned them).
        """
        print(f"  📊 Testing FNR on {len(positive_samples)} positive samples...")
        
        false_negatives = sum(1 for item in positive_samples if not filter_obj.query(item))
        fnr = false_negatives / len(positive_samples)
        
        return fnr
    
    def get_memory_usage(self, filter_obj: Any) -> int:
        """Get memory usage in bytes."""
        if hasattr(filter_obj, 'get_memory_usage'):
            reported = filter_obj.get_memory_usage()
            # Enhanced LBF and some implementations return a dict with
            # a breakdown; prefer the total_bytes entry when present.
            if isinstance(reported, dict):
                return reported.get('total_bytes', 0)
            return reported
        else:
            # Fallback: estimate based on common attributes
            return 1024 * 1024  # 1 MB default
    
    def run_static_workload(self, train_pos: List[Any], train_neg: List[Any],
                           test_pos: List[Any], test_neg: List[Any]) -> Dict[str, Any]:
        """
        Scenario 1: Static Workload
        - Train once on full dataset
        - Query many times
        - Focus: query throughput, memory efficiency, steady-state FPR
        """
        print("\n" + "="*70)
        print("📋 SCENARIO 1: Static Workload (Train Once, Query Many)")
        print("="*70)
        
        results = {}
        
        # Test each implementation
        implementations = {
            'Enhanced LBF': lambda: CombinedEnhancedLBF(
                positive_set=train_pos,
                negative_set=train_neg,
                target_fpr=0.01,
                enable_cache_opt=True,
                enable_incremental=True,
                enable_adaptive=True,
                verbose=False
            ),
            'Adaptive LBF (Ada-BF)': lambda: AdaptiveLearnedBloomFilter(
                positive_set=train_pos,
                negative_set=train_neg,
                target_fpr=0.01,
                n_regions=10,
                verbose=False
            ),
            'Partitioned LBF (PLBF)': lambda: PartitionedLearnedBloomFilter(
                positive_set=train_pos,
                negative_set=train_neg,
                target_fpr=0.01,
                n_partitions=8,
                verbose=False
            ),
            'Stable LBF (s-SLBF)': lambda: StableLearnedBloomFilter(
                positive_set=train_pos,
                negative_set=train_neg,
                target_fpr=0.01,
                retrain_threshold=1000,
                verbose=False
            ),
        }
        
        for name, create_filter in implementations.items():
            print(f"\n🔍 Testing: {name}")
            print("-" * 70)
            
            try:
                # Create filter
                start_time = time.time()
                filter_obj = create_filter()
                construction_time = time.time() - start_time
                
                # Measure metrics
                # Query performance on mixed test set
                query_perf = self.measure_query_performance(filter_obj, test_pos + test_neg, n_iterations=5)
                
                # FPR on unseen negatives (should be low)
                fpr = self.measure_false_positive_rate(filter_obj, test_neg)
                
                # FNR on TRAINING positives (should be 0% - no false negatives on known items)
                fnr_train = self.measure_false_negative_rate(filter_obj, train_pos[:500])
                
                # FNR on TEST positives (may be higher - unseen data)
                fnr_test = self.measure_false_negative_rate(filter_obj, test_pos[:500])
                
                memory = self.get_memory_usage(filter_obj)
                
                # Get filter stats if available
                stats = filter_obj.get_stats() if hasattr(filter_obj, 'get_stats') else {}
                
                results[name] = {
                    'construction_time_sec': construction_time,
                    'query_throughput_ops_per_sec': query_perf['throughput'],
                    'query_latency_mean_ms': query_perf['latency_mean'],
                    'query_latency_p50_ms': query_perf['latency_p50'],
                    'query_latency_p95_ms': query_perf['latency_p95'],
                    'query_latency_p99_ms': query_perf['latency_p99'],
                    'false_positive_rate': fpr,
                    'false_negative_rate_train': fnr_train,
                    'false_negative_rate_test': fnr_test,
                    'memory_bytes': memory,
                    'memory_kb': memory / 1024,
                    'memory_mb': memory / (1024 * 1024),
                    'stats': stats
                }
                
                print(f"  ✅ Construction: {construction_time:.3f}s")
                print(f"  ✅ Query Throughput: {query_perf['throughput']:,.0f} ops/sec")
                print(f"  ✅ Query Latency (mean): {query_perf['latency_mean']:.6f} ms")
                print(f"  ✅ Query Latency (p95): {query_perf['latency_p95']:.6f} ms")
                print(f"  ✅ FPR: {fpr:.4%}")
                print(f"  ✅ FNR (train): {fnr_train:.4%}  (test): {fnr_test:.4%}")
                print(f"  ✅ Memory: {memory / 1024:.2f} KB")
                
                # Cleanup
                del filter_obj
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results[name] = {'error': str(e)}
        
        return results
    
    def run_dynamic_workload(self, train_pos: List[Any], train_neg: List[Any],
                            test_pos: List[Any], test_neg: List[Any]) -> Dict[str, Any]:
        """
        Scenario 2: Dynamic Workload
        - Continuous insertions (streaming)
        - Interleaved queries
        - Focus: update time, FPR stability, throughput under load
        """
        print("\n" + "="*70)
        print("📋 SCENARIO 2: Dynamic Workload (Continuous Insertions)")
        print("="*70)
        
        results = {}
        
        # Use smaller training set, reserve items for streaming insertions
        train_pos_small = train_pos[:len(train_pos)//2]
        train_neg_small = train_neg[:len(train_neg)//2]
        streaming_items = train_pos[len(train_pos)//2:len(train_pos)//2 + 1000]
        
        implementations = {
            'Enhanced LBF': lambda: CombinedEnhancedLBF(
                positive_set=train_pos_small,
                negative_set=train_neg_small,
                target_fpr=0.01,
                enable_incremental=True,
                verbose=False
            ),
            'Stable LBF (s-SLBF)': lambda: StableLearnedBloomFilter(
                positive_set=train_pos_small,
                negative_set=train_neg_small,
                target_fpr=0.01,
                retrain_threshold=500,
                verbose=False
            ),
        }
        
        for name, create_filter in implementations.items():
            print(f"\n🔍 Testing: {name}")
            print("-" * 70)
            
            try:
                filter_obj = create_filter()
                
                # Measure initial FPR
                initial_fpr = self.measure_false_positive_rate(filter_obj, test_neg[:500])
                
                # Perform streaming insertions with interleaved queries
                update_perf = self.measure_update_performance(filter_obj, streaming_items)
                
                # Measure final FPR
                final_fpr = self.measure_false_positive_rate(filter_obj, test_neg[:500])
                
                # Measure query performance after updates
                query_perf = self.measure_query_performance(filter_obj, test_pos[:500], n_iterations=3)
                
                # Calculate FPR stability
                fpr_variance = abs(final_fpr - initial_fpr) / initial_fpr if initial_fpr > 0 else 0
                
                results[name] = {
                    'update_throughput_ops_per_sec': update_perf['throughput'],
                    'update_latency_mean_ms': update_perf['latency_mean'],
                    'query_throughput_after_updates': query_perf['throughput'],
                    'initial_fpr': initial_fpr,
                    'final_fpr': final_fpr,
                    'fpr_variance_percent': fpr_variance * 100,
                    'total_insertions': len(streaming_items)
                }
                
                print(f"  ✅ Update Throughput: {update_perf['throughput']:,.0f} ops/sec")
                print(f"  ✅ Update Latency: {update_perf['latency_mean']:.6f} ms")
                print(f"  ✅ Query Throughput (after updates): {query_perf['throughput']:,.0f} ops/sec")
                print(f"  ✅ Initial FPR: {initial_fpr:.4%}")
                print(f"  ✅ Final FPR: {final_fpr:.4%}")
                print(f"  ✅ FPR Variance: {fpr_variance:.2%}")
                
                del filter_obj
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results[name] = {'error': str(e)}
        
        return results
    
    def run_all_benchmarks(self):
        """Run all benchmark scenarios."""
        print("\n" + "="*70)
        print("🚀 LBF COMPARATIVE BENCHMARK")
        print("="*70)
        
        # Load dataset
        train_pos, train_neg, test_pos, test_neg = self.load_url_dataset(n_samples=5000)
        
        # Run scenarios
        self.results['static_workload'] = self.run_static_workload(
            train_pos, train_neg, test_pos, test_neg
        )
        
        self.results['dynamic_workload'] = self.run_dynamic_workload(
            train_pos, train_neg, test_pos, test_neg
        )
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to JSON file."""
        output_file = self.output_dir / "lbf_comparison_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("📊 BENCHMARK SUMMARY")
        print("="*70)
        
        # Static workload summary
        print("\n🔹 Static Workload (Query Performance):")
        print("-" * 70)
        if 'static_workload' in self.results:
            for name, metrics in self.results['static_workload'].items():
                if 'error' not in metrics:
                    print(f"\n{name}:")
                    print(f"  Query Throughput: {metrics['query_throughput_ops_per_sec']:,.0f} ops/sec")
                    print(f"  Memory: {metrics['memory_kb']:.2f} KB")
                    print(f"  FPR: {metrics['false_positive_rate']:.4%}")
        
        # Dynamic workload summary
        print("\n🔹 Dynamic Workload (Update Performance):")
        print("-" * 70)
        if 'dynamic_workload' in self.results:
            for name, metrics in self.results['dynamic_workload'].items():
                if 'error' not in metrics:
                    print(f"\n{name}:")
                    print(f"  Update Throughput: {metrics['update_throughput_ops_per_sec']:,.0f} ops/sec")
                    print(f"  FPR Variance: ±{metrics['fpr_variance_percent']:.2f}%")


if __name__ == '__main__':
    benchmark = LBFBenchmark()
    benchmark.run_all_benchmarks()
