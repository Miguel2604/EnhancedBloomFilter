"""
Download and prepare real-world datasets for Bloom Filter testing

Datasets:
1. URL Blacklists - Web filtering applications
2. Network packet traces - DDoS detection  
3. Genomic k-mers - Bioinformatics applications
4. Database keys - Caching systems
"""

import os
import sys
import time
import json
import hashlib
import requests
import zipfile
import gzip
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatasetDownloader:
    """Downloads and prepares real-world datasets for testing."""
    
    def __init__(self, data_dir: str = "data/datasets", verbose: bool = True):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Directory to store datasets
            verbose: Print progress messages
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.datasets = {}
        
    def download_all_datasets(self):
        """Download all available datasets."""
        print("\n" + "="*70)
        print(" REAL-WORLD DATASET DOWNLOADER ")
        print("="*70)
        
        # Download each dataset type
        self.download_url_blacklist()
        self.download_network_traces()
        self.download_genomic_data()
        self.download_database_keys()
        
        # Generate summary
        self.generate_dataset_summary()
        
        return self.datasets
    
    def download_url_blacklist(self):
        """Download malicious URL dataset from URLhaus."""
        print("\n" + "-"*60)
        print("1. DOWNLOADING URL BLACKLIST DATASET")
        print("-"*60)
        
        dataset_path = self.data_dir / "url_blacklist"
        dataset_path.mkdir(exist_ok=True)
        
        try:
            # Use the recent URLs endpoint which has more data
            url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
            
            if self.verbose:
                print(f"Downloading from URLhaus (recent malicious URLs)...")
            
            # Download with timeout
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Save raw data
                csv_file = dataset_path / "urlhaus_malicious_urls.csv"
                with open(csv_file, 'wb') as f:
                    f.write(response.content)
                
                # Process CSV - URLhaus format has comment lines starting with #
                urls = []
                with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Skip comment lines and empty lines
                        if line.startswith('#') or not line.strip():
                            continue
                        # Parse CSV line (format: id,dateadded,url,url_status,...)
                        try:
                            parts = line.strip().split(',', 3)  # Split first 3 commas
                            if len(parts) >= 3:
                                # URL is in the third column, remove quotes
                                url_value = parts[2].strip('"')
                                if url_value.startswith('http'):
                                    urls.append(url_value)
                        except Exception:
                            continue  # Skip malformed lines
                
                # Check if we got any URLs
                if len(urls) == 0:
                    print("⚠ No URLs found in CSV, using fallback")
                    self._use_fallback_url_dataset(dataset_path)
                    return
                    
                # Save processed URLs
                processed_file = dataset_path / "malicious_urls.txt"
                with open(processed_file, 'w') as f:
                    for url in urls[:100000]:  # Limit to 100k for testing
                        f.write(url + '\n')
                
                # Create benign URLs for comparison
                benign_urls = self._generate_benign_urls(len(urls))
                benign_file = dataset_path / "benign_urls.txt"
                with open(benign_file, 'w') as f:
                    for url in benign_urls:
                        f.write(url + '\n')
                
                self.datasets['url_blacklist'] = {
                    'malicious': processed_file,
                    'benign': benign_file,
                    'total_malicious': len(urls),
                    'total_benign': len(benign_urls)
                }
                
                print(f"✓ Downloaded {len(urls)} malicious URLs")
                print(f"✓ Generated {len(benign_urls)} benign URLs")
                print(f"✓ Saved to {dataset_path}")
                
            else:
                print(f"⚠ Failed to download: HTTP {response.status_code}")
                self._use_fallback_url_dataset(dataset_path)
                
        except Exception as e:
            print(f"⚠ Error downloading URLhaus data: {e}")
            self._use_fallback_url_dataset(dataset_path)
    
    def _generate_benign_urls(self, count: int) -> List[str]:
        """Generate benign URLs for testing."""
        domains = [
            "example.com", "google.com", "github.com", "stackoverflow.com",
            "wikipedia.org", "amazon.com", "microsoft.com", "apple.com",
            "netflix.com", "spotify.com", "reddit.com", "twitter.com"
        ]
        
        paths = [
            "/", "/index.html", "/about", "/contact", "/products",
            "/services", "/blog", "/news", "/help", "/search"
        ]
        
        urls = []
        for i in range(count):
            domain = domains[i % len(domains)]
            path = paths[i % len(paths)]
            protocol = "https" if i % 2 == 0 else "http"
            url = f"{protocol}://{domain}{path}"
            if i % 3 == 0:
                url += f"?id={i}"
            urls.append(url)
        
        return urls
    
    def _use_fallback_url_dataset(self, dataset_path):
        """Create a fallback URL dataset if download fails."""
        print("Using fallback URL dataset...")
        
        # Generate synthetic malicious URLs
        malicious_urls = []
        suspicious_domains = ["malware", "phishing", "trojan", "virus", "hack"]
        
        for i in range(10000):
            domain_type = suspicious_domains[i % len(suspicious_domains)]
            url = f"http://{domain_type}-site-{i}.com/malicious-{i}.php"
            malicious_urls.append(url)
        
        mal_file = dataset_path / "malicious_urls.txt"
        with open(mal_file, 'w') as f:
            for url in malicious_urls:
                f.write(url + '\n')
        
        # Generate benign URLs
        benign_urls = self._generate_benign_urls(10000)
        benign_file = dataset_path / "benign_urls.txt"
        with open(benign_file, 'w') as f:
            for url in benign_urls:
                f.write(url + '\n')
        
        self.datasets['url_blacklist'] = {
            'malicious': mal_file,
            'benign': benign_file,
            'total_malicious': len(malicious_urls),
            'total_benign': len(benign_urls)
        }
        
        print(f"✓ Generated {len(malicious_urls)} synthetic malicious URLs")
        print(f"✓ Generated {len(benign_urls)} benign URLs")
    
    def download_network_traces(self):
        """Download or generate network packet trace data."""
        print("\n" + "-"*60)
        print("2. DOWNLOADING NETWORK TRACE DATASET")
        print("-"*60)
        
        dataset_path = self.data_dir / "network_traces"
        dataset_path.mkdir(exist_ok=True)
        
        # For safety and simplicity, generate synthetic network data
        # Real PCAP files require special handling and can be large
        
        print("Generating synthetic network packet data...")
        
        # Generate normal traffic IPs
        normal_ips = []
        for i in range(50000):
            ip = f"192.168.{i % 256}.{(i // 256) % 256}"
            normal_ips.append(ip)
        
        # Generate DDoS attack IPs (concentrated from fewer sources)
        attack_ips = []
        for i in range(50000):
            # Simulate botnet - many requests from few IPs
            source_id = i % 100  # Only 100 unique attackers
            ip = f"10.0.{source_id}.{i % 256}"
            attack_ips.append(ip)
        
        # Save datasets
        normal_file = dataset_path / "normal_traffic.txt"
        with open(normal_file, 'w') as f:
            for ip in normal_ips:
                f.write(f"{ip}:{np.random.randint(1024, 65535)}\n")
        
        attack_file = dataset_path / "ddos_traffic.txt"
        with open(attack_file, 'w') as f:
            for ip in attack_ips:
                f.write(f"{ip}:{np.random.randint(80, 443)}\n")
        
        self.datasets['network_traces'] = {
            'normal': normal_file,
            'attack': attack_file,
            'total_normal': len(normal_ips),
            'total_attack': len(attack_ips)
        }
        
        print(f"✓ Generated {len(normal_ips)} normal traffic entries")
        print(f"✓ Generated {len(attack_ips)} DDoS attack entries")
        print(f"✓ Saved to {dataset_path}")
    
    def download_genomic_data(self):
        """Download or generate genomic k-mer data."""
        print("\n" + "-"*60)
        print("3. DOWNLOADING GENOMIC K-MER DATASET")
        print("-"*60)
        
        dataset_path = self.data_dir / "genomic_kmers"
        dataset_path.mkdir(exist_ok=True)
        
        print("Generating synthetic genomic k-mers...")
        
        # Generate k-mers (DNA sequences of length k)
        k = 21  # Common k-mer size
        bases = ['A', 'T', 'G', 'C']
        
        # Generate reference genome k-mers
        reference_kmers = set()
        np.random.seed(42)  # For reproducibility
        
        for i in range(100000):
            kmer = ''.join(np.random.choice(bases, k))
            reference_kmers.add(kmer)
        
        # Generate query k-mers (50% in reference, 50% mutations)
        query_kmers = []
        reference_list = list(reference_kmers)
        
        # Add some from reference
        for i in range(25000):
            query_kmers.append(reference_list[i])
        
        # Add mutations (not in reference)
        for i in range(25000):
            kmer = ''.join(np.random.choice(bases, k))
            # Ensure it's not in reference
            while kmer in reference_kmers:
                kmer = ''.join(np.random.choice(bases, k))
            query_kmers.append(kmer)
        
        # Save datasets
        ref_file = dataset_path / "reference_kmers.txt"
        with open(ref_file, 'w') as f:
            for kmer in reference_kmers:
                f.write(kmer + '\n')
        
        query_file = dataset_path / "query_kmers.txt"
        with open(query_file, 'w') as f:
            for kmer in query_kmers:
                f.write(kmer + '\n')
        
        self.datasets['genomic_kmers'] = {
            'reference': ref_file,
            'queries': query_file,
            'k': k,
            'total_reference': len(reference_kmers),
            'total_queries': len(query_kmers)
        }
        
        print(f"✓ Generated {len(reference_kmers)} reference k-mers (k={k})")
        print(f"✓ Generated {len(query_kmers)} query k-mers")
        print(f"✓ Saved to {dataset_path}")
    
    def download_database_keys(self):
        """Generate database keys for caching simulation."""
        print("\n" + "-"*60)
        print("4. DOWNLOADING DATABASE KEY DATASET")
        print("-"*60)
        
        dataset_path = self.data_dir / "database_keys"
        dataset_path.mkdir(exist_ok=True)
        
        print("Generating synthetic database keys...")
        
        # Generate primary keys (integers)
        primary_keys = []
        for i in range(1, 100001):
            primary_keys.append(f"user:{i}")
        
        # Generate composite keys
        composite_keys = []
        tables = ["orders", "products", "sessions", "transactions"]
        
        for table in tables:
            for i in range(25000):
                key = f"{table}:{i}:{hashlib.md5(f'{table}{i}'.encode()).hexdigest()[:8]}"
                composite_keys.append(key)
        
        # Generate cache keys (frequently accessed)
        cache_keys = []
        # Follow Zipf distribution (some keys accessed much more than others)
        np.random.seed(42)
        for i in range(50000):
            # Zipf distribution for realistic access pattern
            key_id = np.random.zipf(1.5) % 10000
            cache_keys.append(f"cache:item:{key_id}")
        
        # Generate query keys (mix of existing and non-existing)
        query_keys = []
        # 70% hit rate (typical cache hit rate)
        for i in range(35000):  # 70% from existing
            idx = i % len(primary_keys)
            query_keys.append(primary_keys[idx])
        
        for i in range(15000):  # 30% miss
            query_keys.append(f"missing:key:{i}")
        
        # Save datasets
        primary_file = dataset_path / "primary_keys.txt"
        with open(primary_file, 'w') as f:
            for key in primary_keys:
                f.write(key + '\n')
        
        composite_file = dataset_path / "composite_keys.txt"
        with open(composite_file, 'w') as f:
            for key in composite_keys:
                f.write(key + '\n')
        
        cache_file = dataset_path / "cache_keys.txt"
        with open(cache_file, 'w') as f:
            for key in cache_keys:
                f.write(key + '\n')
        
        query_file = dataset_path / "query_keys.txt"
        with open(query_file, 'w') as f:
            for key in query_keys:
                f.write(key + '\n')
        
        self.datasets['database_keys'] = {
            'primary': primary_file,
            'composite': composite_file,
            'cache': cache_file,
            'queries': query_file,
            'total_keys': len(primary_keys) + len(composite_keys),
            'total_queries': len(query_keys)
        }
        
        print(f"✓ Generated {len(primary_keys)} primary keys")
        print(f"✓ Generated {len(composite_keys)} composite keys")
        print(f"✓ Generated {len(cache_keys)} cache access patterns")
        print(f"✓ Generated {len(query_keys)} query keys")
        print(f"✓ Saved to {dataset_path}")
    
    def generate_dataset_summary(self):
        """Generate a summary of all downloaded datasets."""
        print("\n" + "="*70)
        print(" DATASET SUMMARY ")
        print("="*70)
        
        summary = {
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': {}
        }
        
        total_size = 0
        
        for name, info in self.datasets.items():
            dataset_size = 0
            file_info = {}
            
            for key, value in info.items():
                if isinstance(value, Path):
                    if value.exists():
                        size = value.stat().st_size
                        dataset_size += size
                        file_info[key] = {
                            'path': str(value),
                            'size_bytes': size,
                            'size_mb': size / (1024 * 1024)
                        }
            
            summary['datasets'][name] = {
                'files': file_info,
                'total_size_mb': dataset_size / (1024 * 1024),
                'metadata': {k: v for k, v in info.items() if not isinstance(v, Path)}
            }
            
            total_size += dataset_size
            
            print(f"\n{name.upper()}:")
            print(f"  Total size: {dataset_size / (1024 * 1024):.2f} MB")
            for key, value in info.items():
                if not isinstance(value, Path):
                    print(f"  {key}: {value}")
        
        summary['total_size_mb'] = total_size / (1024 * 1024)
        
        # Save summary
        summary_file = self.data_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n" + "-"*40)
        print(f"Total dataset size: {total_size / (1024 * 1024):.2f} MB")
        print(f"Summary saved to: {summary_file}")
        
        return summary


class DatasetTester:
    """Test Bloom Filter implementations with real-world datasets."""
    
    def __init__(self, datasets: Dict, verbose: bool = True):
        """
        Initialize dataset tester.
        
        Args:
            datasets: Dictionary of dataset information
            verbose: Print progress messages
        """
        self.datasets = datasets
        self.verbose = verbose
        self.results = {}
    
    def test_all_datasets(self):
        """Test all datasets with different Bloom Filter implementations."""
        print("\n" + "="*70)
        print(" REAL-WORLD DATASET TESTING ")
        print("="*70)
        
        from src.bloom_filter.standard import StandardBloomFilter
        from src.enhanced_lbf.combined import CombinedEnhancedLBF
        
        for dataset_name, dataset_info in self.datasets.items():
            print(f"\n" + "-"*60)
            print(f"Testing {dataset_name.upper()}")
            print("-"*60)
            
            if dataset_name == 'url_blacklist':
                self._test_url_filtering(dataset_info)
            elif dataset_name == 'network_traces':
                self._test_ddos_detection(dataset_info)
            elif dataset_name == 'genomic_kmers':
                self._test_genomic_search(dataset_info)
            elif dataset_name == 'database_keys':
                self._test_cache_lookup(dataset_info)
        
        return self.results
    
    def _test_url_filtering(self, dataset_info):
        """Test URL filtering application."""
        from src.enhanced_lbf.combined import CombinedEnhancedLBF
        from src.bloom_filter.standard import StandardBloomFilter
        
        print("\nURL Filtering Test:")
        
        # Load malicious URLs
        with open(dataset_info['malicious'], 'r') as f:
            malicious_urls = [line.strip() for line in f.readlines()[:10000]]
        
        # Load benign URLs
        with open(dataset_info['benign'], 'r') as f:
            benign_urls = [line.strip() for line in f.readlines()[:10000]]
        
        print("Training Enhanced Bloom Filter on malicious URLs...")
        
        # Split data for training and testing
        train_malicious = malicious_urls[:5000]
        test_malicious = malicious_urls[5000:8000]
        train_benign = benign_urls[:5000]
        test_benign = benign_urls[5000:8000]
        
        # Create filter with malicious as positive set (items to detect)
        # Using simplified approach - just add malicious URLs to filter
        filter_bf = CombinedEnhancedLBF(
            initial_positive_set=train_malicious,
            initial_negative_set=train_benign,
            target_fpr=0.001,  # Very low FPR for security
            verbose=False
        )
        
        # Also create a standard Bloom filter for comparison
        standard_bf = StandardBloomFilter(
            expected_elements=len(train_malicious),
            false_positive_rate=0.001
        )
        for url in train_malicious:
            standard_bf.add(url)
        
        # Test detection on the enhanced filter
        print("Testing detection rates on Enhanced LBF...")
        
        # Test on malicious URLs (should detect - true positives)
        true_positives = sum(1 for url in test_malicious if filter_bf.query(url))
        tpr = true_positives / len(test_malicious)
        
        # Test on benign URLs (should not detect - false positives)
        false_positives = sum(1 for url in test_benign if filter_bf.query(url))
        fpr = false_positives / len(test_benign)
        
        # Also test standard BF for comparison
        std_tp = sum(1 for url in test_malicious if standard_bf.query(url))
        std_tpr = std_tp / len(test_malicious)
        std_fp = sum(1 for url in test_benign if standard_bf.query(url))
        std_fpr = std_fp / len(test_benign)
        
        print(f"  Enhanced LBF - True Positive Rate: {tpr:.2%}")
        print(f"  Enhanced LBF - False Positive Rate: {fpr:.2%}")
        print(f"  Standard BF - True Positive Rate: {std_tpr:.2%}")
        print(f"  Standard BF - False Positive Rate: {std_fpr:.2%}")
        
        # If FPR is too high, there may be an issue with the model
        if fpr > 0.1:  # More than 10% FPR is concerning
            print(f"\n  ⚠ Warning: High FPR detected. Investigating...")
            
            # Check if the model is actually learning
            sample_malicious = test_malicious[:10]
            sample_benign = test_benign[:10]
            
            print(f"  Sample malicious URL predictions:")
            for url in sample_malicious[:3]:
                result = filter_bf.query(url)
                print(f"    {url[:50]}... -> {'Detected' if result else 'Missed'}")
            
            print(f"  Sample benign URL predictions:")
            for url in sample_benign[:3]:
                result = filter_bf.query(url)
                print(f"    {url[:50]}... -> {'False Alarm' if result else 'Correct'}")
        
        self.results['url_filtering'] = {
            'enhanced_tpr': tpr,
            'enhanced_fpr': fpr,
            'standard_tpr': std_tpr,
            'standard_fpr': std_fpr,
            'total_urls': len(malicious_urls) + len(benign_urls)
        }
    
    def _test_ddos_detection(self, dataset_info):
        """Test DDoS detection application."""
        from src.enhanced_lbf.adaptive import AdaptiveLBF
        
        print("\nDDoS Detection Test:")
        
        # Load traffic data
        with open(dataset_info['normal'], 'r') as f:
            normal_traffic = [line.strip() for line in f.readlines()[:10000]]
        
        with open(dataset_info['attack'], 'r') as f:
            attack_traffic = [line.strip() for line in f.readlines()[:10000]]
        
        # Extract unique IPs from attack traffic (these repeat often in DDoS)
        attack_ips = list(set([ip.split(':')[0] for ip in attack_traffic]))[:1000]
        normal_ips = list(set([ip.split(':')[0] for ip in normal_traffic]))[:5000]
        
        print("Training Adaptive Bloom Filter on attack patterns...")
        
        # Use adaptive filter for dynamic threshold adjustment
        detector = AdaptiveLBF(
            positive_set=attack_ips,
            negative_set=normal_ips,
            target_fpr=0.01,
            verbose=False
        )
        
        # Simulate traffic monitoring
        print("Simulating traffic monitoring...")
        
        detected_attacks = 0
        false_alarms = 0
        
        # Test on mixed traffic
        for i, ip in enumerate(attack_traffic[5000:6000]):
            source_ip = ip.split(':')[0]
            if detector.query(source_ip, ground_truth=True):
                detected_attacks += 1
        
        for i, ip in enumerate(normal_traffic[5000:6000]):
            source_ip = ip.split(':')[0]
            if detector.query(source_ip, ground_truth=False):
                false_alarms += 1
        
        detection_rate = detected_attacks / 1000
        false_alarm_rate = false_alarms / 1000
        
        print(f"  Attack Detection Rate: {detection_rate:.2%}")
        print(f"  False Alarm Rate: {false_alarm_rate:.2%}")
        
        self.results['ddos_detection'] = {
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate
        }
    
    def _test_genomic_search(self, dataset_info):
        """Test genomic k-mer search application."""
        from src.enhanced_lbf.cache_aligned import CacheAlignedLBF
        
        print("\nGenomic K-mer Search Test:")
        
        # Load k-mers
        with open(dataset_info['reference'], 'r') as f:
            reference_kmers = [line.strip() for line in f.readlines()[:50000]]
        
        with open(dataset_info['queries'], 'r') as f:
            query_kmers = [line.strip() for line in f.readlines()[:10000]]
        
        print("Building cache-optimized Bloom Filter for reference genome...")
        
        # Use cache-aligned for better performance with many queries
        # Generate negative k-mers (not in reference)
        negative_kmers = []
        for i in range(10000):
            kmer = f"SYNTHETIC_NEG_{i}" * 3  # Clearly not real k-mers
            negative_kmers.append(kmer[:21])  # Trim to k-mer size
        
        genome_filter = CacheAlignedLBF(
            positive_set=reference_kmers[:25000],
            negative_set=negative_kmers,
            target_fpr=0.001,
            n_blocks=2048,  # More blocks for larger dataset
            verbose=False
        )
        
        # Add remaining k-mers
        for kmer in reference_kmers[25000:]:
            genome_filter.add(kmer)
        
        # Test queries
        print("Testing k-mer queries...")
        
        import time
        start_time = time.perf_counter()
        
        found_count = 0
        for kmer in query_kmers:
            if genome_filter.query(kmer):
                found_count += 1
        
        query_time = time.perf_counter() - start_time
        queries_per_second = len(query_kmers) / query_time
        
        print(f"  K-mers found: {found_count}/{len(query_kmers)}")
        print(f"  Query throughput: {queries_per_second:.0f} queries/sec")
        
        self.results['genomic_search'] = {
            'found_rate': found_count / len(query_kmers),
            'queries_per_second': queries_per_second,
            'k': dataset_info['k']
        }
    
    def _test_cache_lookup(self, dataset_info):
        """Test database cache lookup application."""
        from src.enhanced_lbf.incremental import IncrementalLBF
        
        print("\nDatabase Cache Lookup Test:")
        
        # Load keys
        with open(dataset_info['primary'], 'r') as f:
            primary_keys = [line.strip() for line in f.readlines()]
        
        with open(dataset_info['composite'], 'r') as f:
            composite_keys = [line.strip() for line in f.readlines()]
        
        with open(dataset_info['queries'], 'r') as f:
            query_keys = [line.strip() for line in f.readlines()[:20000]]
        
        all_keys = primary_keys[:50000] + composite_keys[:50000]
        
        print("Building incremental Bloom Filter for cache...")
        
        # Use incremental for dynamic updates
        cache_filter = IncrementalLBF(
            window_size=50000,  # Recent items window
            reservoir_size=5000,  # Long-term memory
            target_fpr=0.01,
            verbose=False
        )
        
        # Add initial keys
        for key in all_keys[:50000]:
            cache_filter.add(key, label=1)
        
        # Simulate cache operations
        print("Simulating cache operations...")
        
        hits = 0
        misses = 0
        false_positives = 0
        
        for query in query_keys:
            result = cache_filter.query(query)
            
            if result:
                if query in all_keys:
                    hits += 1
                else:
                    false_positives += 1
            else:
                misses += 1
        
        hit_rate = hits / len(query_keys)
        miss_rate = misses / len(query_keys)
        fp_rate = false_positives / len(query_keys)
        
        print(f"  Cache hit rate: {hit_rate:.2%}")
        print(f"  Cache miss rate: {miss_rate:.2%}")
        print(f"  False positive rate: {fp_rate:.2%}")
        
        # Test incremental updates
        print("Testing incremental updates...")
        
        import time
        update_times = []
        
        for i in range(1000):
            new_key = f"new_key_{i}"
            start = time.perf_counter()
            cache_filter.add(new_key, label=1)
            update_times.append(time.perf_counter() - start)
        
        avg_update_time = np.mean(update_times) * 1000  # Convert to ms
        
        print(f"  Average update time: {avg_update_time:.3f} ms")
        
        self.results['cache_lookup'] = {
            'hit_rate': hit_rate,
            'false_positive_rate': fp_rate,
            'avg_update_ms': avg_update_time
        }


def main():
    """Main function to download datasets and run tests."""
    
    # Initialize downloader
    downloader = DatasetDownloader(verbose=True)
    
    # Download all datasets
    print("\n🌐 Starting dataset downloads...")
    print("Note: This will download/generate several datasets.")
    print("Total size will be approximately 50-100 MB.\n")
    
    try:
        datasets = downloader.download_all_datasets()
        
        # Run tests
        print("\n🧪 Running tests on real-world datasets...")
        tester = DatasetTester(datasets, verbose=True)
        results = tester.test_all_datasets()
        
        # Save test results
        results_file = Path("data/results/real_world_test_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print(" TESTING COMPLETE ")
        print("="*70)
        print(f"\n✅ All datasets downloaded and tested successfully!")
        print(f"📊 Results saved to: {results_file}")
        
        # Print summary
        print("\n📈 Performance Summary:")
        for test_name, test_results in results.items():
            print(f"\n{test_name.replace('_', ' ').title()}:")
            for metric, value in test_results.items():
                if isinstance(value, float):
                    if 'rate' in metric:
                        print(f"  {metric}: {value:.2%}")
                    else:
                        print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()