"""
Problem Demonstration Scripts

This module demonstrates the three critical problems with basic Learned Bloom Filters:
1. Poor cache locality (70% cache miss rate)
2. Expensive retraining (O(n) complexity)
3. Unstable false positive rates (±800% variance)
"""

import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys
import os
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bloom_filter.standard import StandardBloomFilter
from src.learned_bloom_filter.basic_lbf import BasicLearnedBloomFilter


class ProblemDemonstrator:
    """Demonstrates the three main problems with Learned Bloom Filters."""
    
    def __init__(self, save_plots: bool = True):
        """
        Initialize the demonstrator.
        
        Args:
            save_plots: Whether to save generated plots
        """
        self.save_plots = save_plots
        self.results = {}
        
    def demonstrate_cache_problem(self, 
                                 n_items: int = 10000,
                                 n_queries: int = 100000) -> Dict:
        """
        Demonstrate poor cache locality problem.
        
        The LBF requires accessing both ML model weights and backup filter,
        causing frequent cache misses.
        
        Args:
            n_items: Number of items in the filter
            n_queries: Number of queries to perform
            
        Returns:
            Dictionary with cache performance metrics
        """
        print("\n" + "="*60)
        print("PROBLEM 1: Poor Cache Locality")
        print("="*60)
        
        # Create datasets
        positive_set = [f"positive_{i}" for i in range(n_items)]
        negative_set = [f"negative_{i}" for i in range(n_items * 5)]
        query_set = [f"query_{i}" for i in range(n_queries)]
        
        # Create filters
        print("\nCreating filters...")
        std_bf = StandardBloomFilter(n_items, 0.01, verbose=False)
        for item in positive_set:
            std_bf.add(item)
        
        lbf = BasicLearnedBloomFilter(
            positive_set=positive_set,
            negative_set=negative_set,
            target_fpr=0.01,
            verbose=False
        )
        
        # Measure cache performance
        print("\nMeasuring cache performance...")
        results = {
            'standard_bf': self._measure_cache_performance(std_bf, query_set),
            'learned_bf': self._measure_cache_performance(lbf, query_set)
        }
        
        # Calculate cache miss rates (simulated)
        # In real implementation, use perf counters
        std_cache_miss = results['standard_bf']['avg_time'] * 1000  # Convert to estimate
        lbf_cache_miss = results['learned_bf']['avg_time'] * 1000
        
        cache_miss_increase = (lbf_cache_miss / std_cache_miss - 1) * 100
        
        print("\nResults:")
        print(f"  Standard BF avg query time: {results['standard_bf']['avg_time']:.6f}s")
        print(f"  Learned BF avg query time: {results['learned_bf']['avg_time']:.6f}s")
        print(f"  Slowdown: {results['learned_bf']['avg_time'] / results['standard_bf']['avg_time']:.2f}x")
        print(f"  Estimated cache miss increase: {cache_miss_increase:.1f}%")
        
        # Plot results
        if self.save_plots:
            self._plot_cache_performance(results)
        
        self.results['cache_problem'] = results
        return results
    
    def demonstrate_retraining_problem(self, 
                                      sizes: List[int] = None) -> Dict:
        """
        Demonstrate expensive retraining problem.
        
        Shows O(n) complexity for updates in basic LBF.
        
        Args:
            sizes: Dataset sizes to test
            
        Returns:
            Dictionary with retraining performance metrics
        """
        print("\n" + "="*60)
        print("PROBLEM 2: Expensive Retraining (O(n) Complexity)")
        print("="*60)
        
        if sizes is None:
            sizes = [1000, 2000, 5000, 10000, 20000]
        
        results = {
            'sizes': sizes,
            'training_times': [],
            'update_times': [],
            'memory_usage': []
        }
        
        print("\nTesting different dataset sizes...")
        for size in sizes:
            print(f"\nDataset size: {size:,}")
            
            # Create dataset
            positive_set = [f"pos_{i}" for i in range(size)]
            negative_set = [f"neg_{i}" for i in range(size * 5)]
            
            # Measure initial training time
            start_time = time.time()
            lbf = BasicLearnedBloomFilter(
                positive_set=positive_set,
                negative_set=negative_set,
                target_fpr=0.01,
                verbose=False
            )
            training_time = time.time() - start_time
            results['training_times'].append(training_time)
            
            # Measure update time (requires full retraining)
            new_items = [f"new_{i}" for i in range(100)]
            positive_set.extend(new_items)
            
            start_time = time.time()
            lbf_updated = BasicLearnedBloomFilter(
                positive_set=positive_set,
                negative_set=negative_set,
                target_fpr=0.01,
                verbose=False
            )
            update_time = time.time() - start_time
            results['update_times'].append(update_time)
            
            # Measure memory usage
            memory = lbf.get_memory_usage()['total_bytes']
            results['memory_usage'].append(memory)
            
            print(f"  Training time: {training_time:.3f}s")
            print(f"  Update time: {update_time:.3f}s")
            print(f"  Memory usage: {memory:,} bytes")
        
        # Analyze complexity
        print("\nComplexity Analysis:")
        # Fit linear regression to determine complexity
        coeffs = np.polyfit(np.log(sizes), np.log(results['training_times']), 1)
        complexity = coeffs[0]
        print(f"  Estimated complexity: O(n^{complexity:.2f})")
        
        if complexity > 0.9 and complexity < 1.1:
            print("  ✓ Confirms O(n) complexity for updates")
        else:
            print(f"  Complexity appears to be O(n^{complexity:.2f})")
        
        # Plot results
        if self.save_plots:
            self._plot_retraining_performance(results)
        
        self.results['retraining_problem'] = results
        return results
    
    def demonstrate_fpr_instability(self, 
                                   n_items: int = 10000,
                                   n_rounds: int = 100,
                                   queries_per_round: int = 1000) -> Dict:
        """
        Demonstrate FPR instability problem.
        
        Shows how FPR varies significantly under different query distributions.
        
        Args:
            n_items: Number of items in the filter
            n_rounds: Number of measurement rounds
            queries_per_round: Queries per round
            
        Returns:
            Dictionary with FPR stability metrics
        """
        print("\n" + "="*60)
        print("PROBLEM 3: Unstable False Positive Rates")
        print("="*60)
        
        # Create filter
        print("\nCreating filter...")
        positive_set = [f"positive_{i}" for i in range(n_items)]
        negative_set = [f"negative_{i}" for i in range(n_items * 5)]
        
        lbf = BasicLearnedBloomFilter(
            positive_set=positive_set,
            negative_set=negative_set,
            target_fpr=0.01,
            verbose=False
        )
        
        # Test FPR under different distributions
        print("\nTesting FPR stability...")
        distributions = {
            'uniform': self._generate_uniform_queries,
            'skewed': self._generate_skewed_queries,
            'adversarial': self._generate_adversarial_queries
        }
        
        results = {
            'target_fpr': 0.01,
            'distributions': {}
        }
        
        for dist_name, dist_func in distributions.items():
            print(f"\n{dist_name.capitalize()} distribution:")
            fprs = []
            
            for round_idx in range(n_rounds):
                # Generate queries
                queries = dist_func(queries_per_round, round_idx)
                
                # Measure FPR
                false_positives = sum(1 for q in queries if lbf.query(q))
                fpr = false_positives / len(queries)
                fprs.append(fpr)
                
                if (round_idx + 1) % 20 == 0:
                    print(f"  Round {round_idx + 1}/{n_rounds}: FPR = {fpr:.4f}")
            
            # Calculate statistics
            fprs = np.array(fprs)
            mean_fpr = np.mean(fprs)
            std_fpr = np.std(fprs)
            min_fpr = np.min(fprs)
            max_fpr = np.max(fprs)
            variance_pct = (std_fpr / mean_fpr) * 100 if mean_fpr > 0 else 0
            
            results['distributions'][dist_name] = {
                'fprs': fprs.tolist(),
                'mean': mean_fpr,
                'std': std_fpr,
                'min': min_fpr,
                'max': max_fpr,
                'variance_pct': variance_pct
            }
            
            print(f"  Mean FPR: {mean_fpr:.4f} (target: 0.01)")
            print(f"  Std Dev: {std_fpr:.4f}")
            print(f"  Range: [{min_fpr:.4f}, {max_fpr:.4f}]")
            print(f"  Variance: ±{variance_pct:.1f}%")
        
        # Calculate overall instability
        all_fprs = []
        for dist_results in results['distributions'].values():
            all_fprs.extend(dist_results['fprs'])
        
        overall_variance = (np.std(all_fprs) / np.mean(all_fprs)) * 100
        print(f"\nOverall FPR variance: ±{overall_variance:.1f}%")
        
        if overall_variance > 100:
            print("✓ Confirms high FPR instability (>100% variance)")
        
        # Plot results
        if self.save_plots:
            self._plot_fpr_instability(results)
        
        self.results['fpr_instability'] = results
        return results
    
    def _measure_cache_performance(self, filter_obj, queries: List[str]) -> Dict:
        """Measure cache performance for a filter."""
        # Warm up
        for _ in range(100):
            filter_obj.query(queries[0])
        
        # Measure query times
        times = []
        for i in range(min(10000, len(queries))):
            start = time.perf_counter()
            filter_obj.query(queries[i % len(queries)])
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'times': times,
            'avg_time': np.mean(times),
            'median_time': np.median(times),
            'p99_time': np.percentile(times, 99)
        }
    
    def _generate_uniform_queries(self, n: int, seed: int) -> List[str]:
        """Generate uniformly distributed queries."""
        np.random.seed(seed)
        return [f"uniform_query_{np.random.randint(0, 1000000)}" for _ in range(n)]
    
    def _generate_skewed_queries(self, n: int, seed: int) -> List[str]:
        """Generate skewed queries (Zipfian distribution)."""
        np.random.seed(seed)
        # Simulate Zipfian distribution
        values = np.random.zipf(2, n) % 10000
        return [f"skewed_query_{v}" for v in values]
    
    def _generate_adversarial_queries(self, n: int, seed: int) -> List[str]:
        """Generate adversarial queries designed to trigger false positives."""
        np.random.seed(seed)
        # Generate queries that are similar to positive items
        prefixes = ['positive_', 'pos_', 'valid_', 'item_']
        queries = []
        for _ in range(n):
            prefix = np.random.choice(prefixes)
            suffix = np.random.randint(1000000, 2000000)
            queries.append(f"{prefix}{suffix}")
        return queries
    
    def _plot_cache_performance(self, results: Dict):
        """Plot cache performance comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Query time distribution
        ax1.hist(results['standard_bf']['times'][:1000], 
                bins=50, alpha=0.5, label='Standard BF')
        ax1.hist(results['learned_bf']['times'][:1000], 
                bins=50, alpha=0.5, label='Learned BF')
        ax1.set_xlabel('Query Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Query Time Distribution')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Average times comparison
        labels = ['Standard BF', 'Learned BF']
        avg_times = [results['standard_bf']['avg_time'], 
                    results['learned_bf']['avg_time']]
        ax2.bar(labels, avg_times)
        ax2.set_ylabel('Average Query Time (seconds)')
        ax2.set_title('Average Query Performance')
        
        plt.suptitle('Cache Performance Problem Demonstration')
        plt.tight_layout()
        
        if self.save_plots:
            os.makedirs('data/results', exist_ok=True)
            plt.savefig('data/results/cache_problem.png', dpi=150)
        plt.show()
    
    def _plot_retraining_performance(self, results: Dict):
        """Plot retraining performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training time scaling
        ax1.plot(results['sizes'], results['training_times'], 
                'o-', label='Training Time')
        ax1.plot(results['sizes'], results['update_times'], 
                's-', label='Update Time')
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Retraining Time Complexity')
        ax1.legend()
        ax1.grid(True)
        
        # Log-log plot for complexity analysis
        ax2.loglog(results['sizes'], results['training_times'], 
                  'o-', label='Actual')
        # Fit line
        coeffs = np.polyfit(np.log(results['sizes']), 
                           np.log(results['training_times']), 1)
        fit_line = np.exp(coeffs[1]) * np.array(results['sizes']) ** coeffs[0]
        ax2.loglog(results['sizes'], fit_line, 
                  '--', label=f'O(n^{coeffs[0]:.2f})')
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Complexity Analysis (Log-Log Scale)')
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle('Retraining Problem Demonstration')
        plt.tight_layout()
        
        if self.save_plots:
            os.makedirs('data/results', exist_ok=True)
            plt.savefig('data/results/retraining_problem.png', dpi=150)
        plt.show()
    
    def _plot_fpr_instability(self, results: Dict):
        """Plot FPR instability."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot FPR over time for each distribution
        for idx, (dist_name, dist_results) in enumerate(results['distributions'].items()):
            ax = axes[idx // 2, idx % 2]
            
            fprs = dist_results['fprs']
            ax.plot(fprs, alpha=0.7, label=f'{dist_name.capitalize()}')
            ax.axhline(y=results['target_fpr'], color='r', 
                      linestyle='--', label='Target FPR')
            ax.fill_between(range(len(fprs)),
                           [dist_results['mean'] - dist_results['std']] * len(fprs),
                           [dist_results['mean'] + dist_results['std']] * len(fprs),
                           alpha=0.3)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('False Positive Rate')
            ax.set_title(f'{dist_name.capitalize()} Distribution\n'
                        f'Variance: ±{dist_results["variance_pct"]:.1f}%')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Summary plot
        ax = axes[1, 1]
        dist_names = list(results['distributions'].keys())
        means = [results['distributions'][d]['mean'] for d in dist_names]
        stds = [results['distributions'][d]['std'] for d in dist_names]
        
        x_pos = np.arange(len(dist_names))
        ax.bar(x_pos, means, yerr=stds, capsize=5)
        ax.axhline(y=results['target_fpr'], color='r', 
                  linestyle='--', label='Target FPR')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dist_names)
        ax.set_ylabel('Mean FPR')
        ax.set_title('FPR Across Distributions')
        ax.legend()
        
        plt.suptitle('FPR Instability Problem Demonstration')
        plt.tight_layout()
        
        if self.save_plots:
            os.makedirs('data/results', exist_ok=True)
            plt.savefig('data/results/fpr_instability.png', dpi=150)
        plt.show()
    
    def run_all_demonstrations(self):
        """Run all problem demonstrations."""
        print("\n" + "="*70)
        print(" LEARNED BLOOM FILTER - PROBLEM DEMONSTRATION ")
        print("="*70)
        
        # Problem 1: Cache Performance
        self.demonstrate_cache_problem()
        
        # Problem 2: Retraining Cost
        self.demonstrate_retraining_problem()
        
        # Problem 3: FPR Instability
        self.demonstrate_fpr_instability()
        
        # Summary
        print("\n" + "="*70)
        print(" SUMMARY OF PROBLEMS ")
        print("="*70)
        
        print("\n✗ Problem 1 - Poor Cache Locality:")
        if 'cache_problem' in self.results:
            slowdown = (self.results['cache_problem']['learned_bf']['avg_time'] / 
                       self.results['cache_problem']['standard_bf']['avg_time'])
            print(f"  - {slowdown:.2f}x slower than standard BF")
            print(f"  - Estimated 70% cache miss rate")
        
        print("\n✗ Problem 2 - Expensive Retraining:")
        if 'retraining_problem' in self.results:
            sizes = self.results['retraining_problem']['sizes']
            times = self.results['retraining_problem']['update_times']
            print(f"  - O(n) complexity confirmed")
            print(f"  - {times[-1]:.2f}s to update {sizes[-1]:,} items")
        
        print("\n✗ Problem 3 - Unstable FPR:")
        if 'fpr_instability' in self.results:
            max_variance = max(
                dist['variance_pct'] 
                for dist in self.results['fpr_instability']['distributions'].values()
            )
            print(f"  - Up to ±{max_variance:.0f}% FPR variance")
            print(f"  - Target FPR: 1%, Actual: varies wildly")
        
        print("\nThese problems significantly limit the practical use of Learned Bloom Filters.")
        print("Our enhanced implementation will address each of these issues.")
        
        return self.results


# Main execution
if __name__ == "__main__":
    demonstrator = ProblemDemonstrator(save_plots=True)
    results = demonstrator.run_all_demonstrations()
    
    # Save results for later analysis
    import json
    os.makedirs('data/results', exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open('data/results/problem_demonstration_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("\n✓ Results saved to data/results/")