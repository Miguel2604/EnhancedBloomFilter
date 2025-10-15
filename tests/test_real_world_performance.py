"""
Comprehensive real-world dataset testing for all Bloom Filter implementations.

This script tests all implementations with real-world datasets to verify:
1. Correctness - No false negatives
2. Performance - Query speed, memory usage
3. FPR accuracy - Actual vs target FPR
4. Scalability - Performance with different dataset sizes
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all implementations
from src.bloom_filter.standard import StandardBloomFilter
from src.learned_bloom_filter.basic_lbf import BasicLearnedBloomFilter
from src.enhanced_lbf.adaptive import AdaptiveLBF
from src.enhanced_lbf.cache_aligned import CacheAlignedLBF
from src.enhanced_lbf.incremental import IncrementalLBF
from src.enhanced_lbf.combined import CombinedEnhancedLBF


class RealWorldTester:
    """Test all Bloom Filter implementations with real-world datasets."""
    
    def __init__(self, verbose: bool = True, save_results: bool = True):
        """
        Initialize the tester.
        
        Args:
            verbose: Whether to print progress
            save_results: Whether to save results to file
        """
        self.verbose = verbose
        self.save_results = save_results
        self.results = {}
        self.datasets = {}
        
    def load_datasets(self) -> Dict:
        """Load all available real-world datasets."""
        print("\n" + "="*70)
        print(" LOADING REAL-WORLD DATASETS ")
        print("="*70)
        
        # Check if datasets exist, if not download them
        dataset_base = "data/datasets"
        if not os.path.exists(dataset_base):
            print("\nDatasets not found. Downloading...")
            from scripts.download_datasets import DatasetDownloader
            downloader = DatasetDownloader(verbose=True)
            downloader.download_all_datasets()
        
        # Load URL blacklist dataset
        self.datasets['url_blacklist'] = self._load_url_dataset()
        
        # Load network traces dataset
        self.datasets['network_traces'] = self._load_network_dataset()
        
        # Load genomic k-mer dataset
        self.datasets['genomic_kmers'] = self._load_genomic_dataset()
        
        # Load database keys dataset
        self.datasets['database_keys'] = self._load_database_dataset()
        
        print("\nDatasets loaded successfully!")
        return self.datasets
    
    def _load_url_dataset(self) -> Dict:
        """Load URL blacklist dataset."""
        print("\nLoading URL blacklist dataset...")
        
        dataset = {
            'name': 'URL Blacklist',
            'malicious': [],
            'benign': [],
            'queries': []
        }
        
        mal_file = "data/datasets/url_blacklist/malicious_urls.txt"
        benign_file = "data/datasets/url_blacklist/benign_urls.txt"
        
        if os.path.exists(mal_file):
            with open(mal_file, 'r') as f:
                dataset['malicious'] = [line.strip() for line in f.readlines() 
                                       if line.strip()][:50000]
        
        if os.path.exists(benign_file):
            with open(benign_file, 'r') as f:
                all_benign = [line.strip() for line in f.readlines() 
                             if line.strip()]
                dataset['benign'] = all_benign[:25000]
                dataset['queries'] = all_benign[25000:35000]  # Test queries
        
        print(f"  Loaded {len(dataset['malicious'])} malicious URLs")
        print(f"  Loaded {len(dataset['benign'])} benign URLs")
        print(f"  Prepared {len(dataset['queries'])} test queries")
        
        return dataset
    
    def _load_network_dataset(self) -> Dict:
        """Load network traces dataset."""
        print("\nLoading network traces dataset...")
        
        dataset = {
            'name': 'Network Traces',
            'attack': [],
            'normal': [],
            'queries': []
        }
        
        attack_file = "data/datasets/network_traces/ddos_traffic.txt"
        normal_file = "data/datasets/network_traces/normal_traffic.txt"
        
        if os.path.exists(attack_file):
            with open(attack_file, 'r') as f:
                dataset['attack'] = [line.strip() for line in f.readlines() 
                                    if line.strip()][:25000]
        
        if os.path.exists(normal_file):
            with open(normal_file, 'r') as f:
                all_normal = [line.strip() for line in f.readlines() 
                             if line.strip()]
                dataset['normal'] = all_normal[:25000]
                dataset['queries'] = all_normal[25000:35000]
        
        print(f"  Loaded {len(dataset['attack'])} attack IPs")
        print(f"  Loaded {len(dataset['normal'])} normal IPs")
        print(f"  Prepared {len(dataset['queries'])} test queries")
        
        return dataset
    
    def _load_genomic_dataset(self) -> Dict:
        """Load genomic k-mer dataset."""
        print("\nLoading genomic k-mer dataset...")
        
        dataset = {
            'name': 'Genomic K-mers',
            'reference': [],
            'queries': [],
            'negative': []
        }
        
        ref_file = "data/datasets/genomic_kmers/reference_kmers.txt"
        query_file = "data/datasets/genomic_kmers/query_kmers.txt"
        
        if os.path.exists(ref_file):
            with open(ref_file, 'r') as f:
                dataset['reference'] = [line.strip() for line in f.readlines() 
                                       if line.strip()][:50000]
        
        if os.path.exists(query_file):
            with open(query_file, 'r') as f:
                all_queries = [line.strip() for line in f.readlines() 
                              if line.strip()]
                dataset['queries'] = all_queries[:10000]
                # Use some queries as negatives (likely not in reference)
                dataset['negative'] = all_queries[25000:35000]
        
        print(f"  Loaded {len(dataset['reference'])} reference k-mers")
        print(f"  Loaded {len(dataset['queries'])} query k-mers")
        
        return dataset
    
    def _load_database_dataset(self) -> Dict:
        """Load database keys dataset."""
        print("\nLoading database keys dataset...")
        
        dataset = {
            'name': 'Database Keys',
            'primary': [],
            'composite': [],
            'cache': [],
            'queries': []
        }
        
        primary_file = "data/datasets/database_keys/primary_keys.txt"
        composite_file = "data/datasets/database_keys/composite_keys.txt"
        cache_file = "data/datasets/database_keys/cache_keys.txt"
        query_file = "data/datasets/database_keys/query_keys.txt"
        
        if os.path.exists(primary_file):
            with open(primary_file, 'r') as f:
                dataset['primary'] = [line.strip() for line in f.readlines() 
                                     if line.strip()][:25000]
        
        if os.path.exists(composite_file):
            with open(composite_file, 'r') as f:
                dataset['composite'] = [line.strip() for line in f.readlines() 
                                       if line.strip()][:25000]
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                dataset['cache'] = [line.strip() for line in f.readlines() 
                                   if line.strip()][:10000]
        
        if os.path.exists(query_file):
            with open(query_file, 'r') as f:
                dataset['queries'] = [line.strip() for line in f.readlines() 
                                     if line.strip()][:10000]
        
        print(f"  Loaded {len(dataset['primary'])} primary keys")
        print(f"  Loaded {len(dataset['composite'])} composite keys")
        print(f"  Loaded {len(dataset['queries'])} query keys")
        
        return dataset
    
    def test_implementation(self, 
                           filter_class,
                           filter_name: str,
                           dataset: Dict,
                           dataset_name: str,
                           target_fpr: float = 0.01) -> Dict:
        """
        Test a single filter implementation on a dataset.
        
        Args:
            filter_class: The filter class to test
            filter_name: Name of the filter
            dataset: Dataset to test on
            dataset_name: Name of the dataset
            target_fpr: Target false positive rate
            
        Returns:
            Dictionary with test results
        """
        print(f"\nTesting {filter_name} on {dataset_name}...")
        
        results = {
            'filter': filter_name,
            'dataset': dataset_name,
            'target_fpr': target_fpr
        }
        
        # Prepare data based on dataset type
        if dataset_name == 'URL Blacklist':
            positive_set = dataset['malicious'][:10000]
            negative_set = dataset['benign'][:10000]
            test_positive = dataset['malicious'][10000:11000]
            test_negative = dataset['queries'][:1000]
        elif dataset_name == 'Network Traces':
            positive_set = dataset['attack'][:10000]
            negative_set = dataset['normal'][:10000]
            test_positive = dataset['attack'][10000:11000]
            test_negative = dataset['queries'][:1000]
        elif dataset_name == 'Genomic K-mers':
            positive_set = dataset['reference'][:10000]
            negative_set = dataset['negative'][:5000] if dataset['negative'] else []
            test_positive = dataset['reference'][10000:11000]
            test_negative = dataset['queries'][:1000]
        else:  # Database Keys
            positive_set = dataset['primary'][:10000] + dataset['composite'][:10000]
            negative_set = []  # Not needed for standard BF
            test_positive = dataset['primary'][10000:11000]
            test_negative = [f"nonexistent_key_{i}" for i in range(1000)]
        
        # Create and train filter
        print(f"  Training on {len(positive_set)} positive items...")
        
        start_time = time.time()
        
        try:
            if filter_name == "Standard BF":
                bf = filter_class(
                    expected_elements=len(positive_set),
                    false_positive_rate=target_fpr
                )
                for item in positive_set:
                    bf.add(item)
            elif filter_name == "Basic LBF":
                bf = filter_class(
                    positive_set=positive_set,
                    negative_set=negative_set if negative_set else positive_set[:1000],
                    target_fpr=target_fpr,
                    verbose=False
                )
            elif filter_name in ["Adaptive LBF", "Cache-Aligned LBF", "Combined LBF"]:
                bf = filter_class(
                    positive_set=positive_set[:5000],  # Start with subset
                    negative_set=negative_set[:5000] if negative_set else positive_set[:1000],
                    target_fpr=target_fpr,
                    verbose=False
                )
                # Add remaining items
                for item in positive_set[5000:]:
                    if hasattr(bf, 'add'):
                        bf.add(item)
            elif filter_name == "Incremental LBF":
                bf = filter_class(
                    window_size=5000,
                    reservoir_size=1000,
                    target_fpr=target_fpr,
                    verbose=False
                )
                # Add items incrementally
                for item in positive_set:
                    bf.add(item, label=1)
            else:
                # Default initialization
                bf = filter_class(
                    positive_set=positive_set,
                    negative_set=negative_set if negative_set else positive_set[:1000],
                    target_fpr=target_fpr,
                    verbose=False
                )
            
            training_time = time.time() - start_time
            results['training_time'] = training_time
            
            # Test for false negatives (should be 0)
            print(f"  Testing for false negatives...")
            false_negatives = 0
            for item in test_positive:
                if not bf.query(item):
                    false_negatives += 1
            
            fnr = false_negatives / len(test_positive) if test_positive else 0
            results['false_negative_rate'] = fnr
            
            # Test for false positives
            print(f"  Testing false positive rate...")
            false_positives = 0
            for item in test_negative:
                if bf.query(item):
                    false_positives += 1
            
            fpr = false_positives / len(test_negative) if test_negative else 0
            results['false_positive_rate'] = fpr
            
            # Measure query performance
            print(f"  Measuring query performance...")
            query_times = []
            test_queries = test_positive[:100] + test_negative[:100]
            
            for query in test_queries:
                start = time.perf_counter()
                bf.query(query)
                end = time.perf_counter()
                query_times.append(end - start)
            
            results['avg_query_time'] = np.mean(query_times)
            results['p99_query_time'] = np.percentile(query_times, 99)
            
            # Measure memory usage
            if hasattr(bf, 'get_memory_usage'):
                memory = bf.get_memory_usage()
                if isinstance(memory, dict):
                    results['memory_bytes'] = memory.get('total_bytes', 0)
                else:
                    results['memory_bytes'] = memory
            else:
                results['memory_bytes'] = 0
            
            results['success'] = True
            
            print(f"  ✓ Training time: {training_time:.3f}s")
            print(f"  ✓ FNR: {fnr:.4f} (should be 0)")
            print(f"  ✓ FPR: {fpr:.4f} (target: {target_fpr})")
            print(f"  ✓ Avg query time: {results['avg_query_time']*1e6:.2f}μs")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def run_all_tests(self):
        """Run all tests on all implementations with all datasets."""
        # Load datasets
        self.load_datasets()
        
        print("\n" + "="*70)
        print(" RUNNING COMPREHENSIVE TESTS ")
        print("="*70)
        
        # Define implementations to test
        implementations = [
            (StandardBloomFilter, "Standard BF"),
            (BasicLearnedBloomFilter, "Basic LBF"),
            (AdaptiveLBF, "Adaptive LBF"),
            (CacheAlignedLBF, "Cache-Aligned LBF"),
            (IncrementalLBF, "Incremental LBF"),
            (CombinedEnhancedLBF, "Combined LBF")
        ]
        
        # Test each implementation on each dataset
        for dataset_name, dataset in self.datasets.items():
            print(f"\n\n" + "-"*60)
            print(f" Testing on {dataset['name']} Dataset")
            print("-"*60)
            
            for filter_class, filter_name in implementations:
                key = f"{filter_name}_{dataset_name}"
                self.results[key] = self.test_implementation(
                    filter_class,
                    filter_name,
                    dataset,
                    dataset['name']
                )
        
        # Generate summary
        self._generate_summary()
        
        # Save results if requested
        if self.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_summary(self):
        """Generate and print a summary of all test results."""
        print("\n" + "="*70)
        print(" TEST SUMMARY ")
        print("="*70)
        
        # Create comparison table
        summary_data = []
        
        for key, result in self.results.items():
            if result.get('success', False):
                summary_data.append({
                    'Implementation': result['filter'],
                    'Dataset': result['dataset'],
                    'FPR': f"{result['false_positive_rate']:.4f}",
                    'FNR': f"{result['false_negative_rate']:.4f}",
                    'Training (s)': f"{result['training_time']:.3f}",
                    'Query (μs)': f"{result['avg_query_time']*1e6:.2f}",
                    'Memory (KB)': f"{result['memory_bytes']/1024:.1f}"
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            print("\n" + df.to_string(index=False))
        
        # Identify best performers
        print("\n" + "-"*60)
        print(" BEST PERFORMERS")
        print("-"*60)
        
        # Best FPR accuracy
        best_fpr = min(
            (r for r in self.results.values() if r.get('success', False)),
            key=lambda x: abs(x['false_positive_rate'] - x['target_fpr'])
        )
        print(f"\n✓ Most Accurate FPR: {best_fpr['filter']}")
        print(f"  Dataset: {best_fpr['dataset']}")
        print(f"  Target FPR: {best_fpr['target_fpr']}, Actual: {best_fpr['false_positive_rate']:.4f}")
        
        # Fastest queries
        fastest = min(
            (r for r in self.results.values() if r.get('success', False)),
            key=lambda x: x['avg_query_time']
        )
        print(f"\n✓ Fastest Queries: {fastest['filter']}")
        print(f"  Dataset: {fastest['dataset']}")
        print(f"  Avg query time: {fastest['avg_query_time']*1e6:.2f}μs")
        
        # Most memory efficient
        smallest = min(
            (r for r in self.results.values() if r.get('success', False) and r['memory_bytes'] > 0),
            key=lambda x: x['memory_bytes']
        )
        print(f"\n✓ Most Memory Efficient: {smallest['filter']}")
        print(f"  Dataset: {smallest['dataset']}")
        print(f"  Memory usage: {smallest['memory_bytes']/1024:.1f} KB")
    
    def _save_results(self):
        """Save test results to file."""
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        output_file = output_dir / "real_world_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        # Generate plots
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate performance comparison plots."""
        # Prepare data for plotting
        implementations = set()
        datasets = set()
        
        for key, result in self.results.items():
            if result.get('success', False):
                implementations.add(result['filter'])
                datasets.add(result['dataset'])
        
        implementations = sorted(list(implementations))
        datasets = sorted(list(datasets))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. FPR Comparison
        ax = axes[0, 0]
        for impl in implementations:
            fprs = []
            ds_names = []
            for ds in datasets:
                key = f"{impl}_{ds.lower().replace(' ', '_')}"
                if key in self.results and self.results[key].get('success', False):
                    fprs.append(self.results[key]['false_positive_rate'])
                    ds_names.append(ds[:10])  # Truncate long names
            
            if fprs:
                x_pos = np.arange(len(ds_names))
                ax.plot(x_pos, fprs, 'o-', label=impl, alpha=0.7)
        
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels([d[:10] for d in datasets], rotation=45)
        ax.set_ylabel('False Positive Rate')
        ax.set_title('FPR Across Datasets')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. Query Time Comparison
        ax = axes[0, 1]
        for impl in implementations:
            times = []
            ds_names = []
            for ds in datasets:
                key = f"{impl}_{ds.lower().replace(' ', '_')}"
                if key in self.results and self.results[key].get('success', False):
                    times.append(self.results[key]['avg_query_time'] * 1e6)  # Convert to μs
                    ds_names.append(ds[:10])
            
            if times:
                x_pos = np.arange(len(ds_names))
                ax.plot(x_pos, times, 's-', label=impl, alpha=0.7)
        
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels([d[:10] for d in datasets], rotation=45)
        ax.set_ylabel('Query Time (μs)')
        ax.set_title('Query Performance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. Memory Usage Comparison
        ax = axes[1, 0]
        for impl in implementations:
            memory = []
            ds_names = []
            for ds in datasets:
                key = f"{impl}_{ds.lower().replace(' ', '_')}"
                if key in self.results and self.results[key].get('success', False):
                    mem_kb = self.results[key]['memory_bytes'] / 1024
                    if mem_kb > 0:
                        memory.append(mem_kb)
                        ds_names.append(ds[:10])
            
            if memory:
                x_pos = np.arange(len(ds_names))
                ax.plot(x_pos, memory, '^-', label=impl, alpha=0.7)
        
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels([d[:10] for d in datasets], rotation=45)
        ax.set_ylabel('Memory (KB)')
        ax.set_title('Memory Usage')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. Training Time Comparison
        ax = axes[1, 1]
        for impl in implementations:
            train_times = []
            ds_names = []
            for ds in datasets:
                key = f"{impl}_{ds.lower().replace(' ', '_')}"
                if key in self.results and self.results[key].get('success', False):
                    train_times.append(self.results[key]['training_time'])
                    ds_names.append(ds[:10])
            
            if train_times:
                x_pos = np.arange(len(ds_names))
                ax.plot(x_pos, train_times, 'd-', label=impl, alpha=0.7)
        
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels([d[:10] for d in datasets], rotation=45)
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Training Performance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Real-World Dataset Performance Comparison', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        output_file = Path("data/results/real_world_performance.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Performance plots saved to {output_file}")


def main():
    """Main function to run all tests."""
    print("\n" + "="*80)
    print(" BLOOM FILTER REAL-WORLD DATASET TESTING ")
    print("="*80)
    
    tester = RealWorldTester(verbose=True, save_results=True)
    results = tester.run_all_tests()
    
    print("\n" + "="*80)
    print(" TESTING COMPLETE ")
    print("="*80)
    
    # Print final summary
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_count = len(results)
    
    print(f"\n✓ Successfully tested {success_count}/{total_count} configurations")
    print(f"✓ Results saved to data/results/real_world_test_results.json")
    print(f"✓ Plots saved to data/results/real_world_performance.png")
    
    return results


if __name__ == "__main__":
    results = main()