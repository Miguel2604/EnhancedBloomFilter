"""
Adaptive Learned Bloom Filter (Ada-BF) Implementation

Reference: Dai & Shrivastava (2019) - "Adaptive Learned Bloom Filter (Ada-BF): 
Efficient Utilization of the Classifier"

Key Innovation: Uses the full spectrum of predicted probability scores instead of 
just binary classification. Partitions the probability space into regions, each 
with its own backup filter strategy.
"""

import numpy as np
from typing import Any, List, Optional, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bloom_filter.standard import StandardBloomFilter


class AdaptiveLearnedBloomFilter:
    """
    Adaptive Learned Bloom Filter that partitions probability space.
    
    The key idea is to divide the predicted probability range [0, 1] into K regions,
    and assign different backup filter sizes based on the expected false positive
    rate in each region.
    """
    
    def __init__(self,
                 positive_set: List[Any],
                 negative_set: List[Any],
                 target_fpr: float = 0.01,
                 n_regions: int = 10,
                 verbose: bool = False):
        """
        Initialize Ada-BF.
        
        Args:
            positive_set: Training positive examples
            negative_set: Training negative examples
            target_fpr: Target false positive rate
            n_regions: Number of probability regions (K in paper)
            verbose: Print initialization details
        """
        self.target_fpr = target_fpr
        self.n_regions = n_regions
        self.verbose = verbose
        
        if not positive_set or not negative_set:
            raise ValueError("Both positive and negative training sets required")
        
        # Store sets
        self.positive_set = set(positive_set)
        self.negative_set = set(negative_set)
        
        # Train model
        self._train_model(positive_set, negative_set)
        
        # Initialize region-specific backup filters
        self._init_backup_filters(positive_set)
        
        # Statistics
        self.total_queries = 0
        self.model_rejections = 0
        self.backup_queries = 0
        
        if verbose:
            print(f"Initialized Ada-BF:")
            print(f"  Positive training examples: {len(positive_set):,}")
            print(f"  Negative training examples: {len(negative_set):,}")
            print(f"  Target FPR: {target_fpr:.4f}")
            print(f"  Number of regions: {n_regions}")
            print(f"  Model memory: ~{self._estimate_model_size() / 1024:.2f} KB")
    
    def _extract_features(self, item: Any) -> np.ndarray:
        """
        Extract features from an item.
        
        For URLs: domain length, path segments, TLD, special chars, etc.
        For general items: hash-based features
        """
        item_str = str(item)
        
        features = []
        
        # Basic string features
        features.append(len(item_str))
        features.append(item_str.count('.'))
        features.append(item_str.count('/'))
        features.append(item_str.count('-'))
        features.append(item_str.count('_'))
        features.append(sum(c.isdigit() for c in item_str))
        features.append(sum(c.isupper() for c in item_str))
        features.append(sum(c.islower() for c in item_str))
        
        # Hash-based features for additional discrimination
        import hashlib
        hash_bytes = hashlib.sha256(item_str.encode()).digest()
        for i in range(0, 12, 2):
            features.append(int.from_bytes(hash_bytes[i:i+2], 'big'))
        
        return np.array(features, dtype=np.float32)
    
    def _train_model(self, positive_set: List[Any], negative_set: List[Any]):
        """Train the ML classifier on positive and negative examples."""
        # Extract features
        X_pos = np.array([self._extract_features(item) for item in positive_set])
        X_neg = np.array([self._extract_features(item) for item in negative_set])
        
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(positive_set) + [0] * len(negative_set))
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SGD classifier with probability estimates
        self.model = SGDClassifier(
            loss='log_loss',  # Enables probability estimates
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            random_state=42,
            warm_start=True
        )
        
        # Train with multiple epochs for better convergence
        for _ in range(3):
            indices = np.random.permutation(len(X))
            self.model.partial_fit(X_scaled[indices], y[indices], classes=[0, 1])
    
    def _init_backup_filters(self, positive_set: List[Any]):
        """
        Initialize region-specific backup filters.
        
        The idea is to predict probabilities for all items, partition them into
        regions, and create appropriately-sized backup filters for each region.
        """
        # Predict probabilities for all positive items
        X_pos = np.array([self._extract_features(item) for item in positive_set])
        X_pos_scaled = self.scaler.transform(X_pos)
        probs = self.model.predict_proba(X_pos_scaled)[:, 1]  # Probability of class 1
        
        # Define region boundaries
        self.region_boundaries = np.linspace(0, 1, self.n_regions + 1)
        
        # Assign items to regions and create backup filters
        self.backup_filters = []
        self.region_items = [[] for _ in range(self.n_regions)]
        
        for item, prob in zip(positive_set, probs):
            region_idx = self._get_region_index(prob)
            self.region_items[region_idx].append(item)
        
        # Create backup filter for each region
        for region_idx, items in enumerate(self.region_items):
            if items:
                # Lower confidence regions get larger backup filters
                region_fpr = self.target_fpr * (1 + (self.n_regions - region_idx) / self.n_regions)
                bf = StandardBloomFilter(
                    expected_elements=max(len(items), 10),
                    false_positive_rate=min(region_fpr, 0.1)
                )
                for item in items:
                    bf.add(item)
                self.backup_filters.append(bf)
            else:
                # Empty region - create minimal filter
                self.backup_filters.append(
                    StandardBloomFilter(expected_elements=10, false_positive_rate=self.target_fpr)
                )
    
    def _get_region_index(self, probability: float) -> int:
        """Determine which region a probability falls into."""
        for i in range(self.n_regions):
            if self.region_boundaries[i] <= probability < self.region_boundaries[i + 1]:
                return i
        return self.n_regions - 1  # Handle edge case for probability = 1.0
    
    def add(self, item: Any):
        """
        Add an item to the filter.
        
        For Ada-BF, we need to predict which region it belongs to and add to that backup.
        """
        features = self._extract_features(item)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.model.predict_proba(features_scaled)[0, 1]
        
        region_idx = self._get_region_index(prob)
        self.backup_filters[region_idx].add(item)
        self.positive_set.add(item)
    
    def query(self, item: Any) -> bool:
        """
        Query whether an item is in the set.
        
        Returns True if likely positive, False if definitely negative.
        """
        self.total_queries += 1
        
        # Extract features and predict probability
        features = self._extract_features(item)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.model.predict_proba(features_scaled)[0, 1]
        
        # Determine region
        region_idx = self._get_region_index(prob)
        
        # Query the appropriate backup filter
        self.backup_queries += 1
        return self.backup_filters[region_idx].query(item)
    
    def _estimate_model_size(self) -> int:
        """Estimate memory usage of the model in bytes."""
        # SGDClassifier: coefficients + intercept
        model_size = 0
        if hasattr(self.model, 'coef_'):
            model_size += self.model.coef_.nbytes
        if hasattr(self.model, 'intercept_'):
            model_size += self.model.intercept_.nbytes
        
        # Scaler parameters
        model_size += self.scaler.mean_.nbytes + self.scaler.scale_.nbytes
        
        return model_size
    
    def get_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        total = self._estimate_model_size()
        
        # Add backup filter sizes
        for bf in self.backup_filters:
            total += bf.m // 8  # Bits to bytes
        
        return total
    
    def get_stats(self) -> dict:
        """Get statistics about filter performance."""
        return {
            'total_queries': self.total_queries,
            'model_rejections': self.model_rejections,
            'backup_queries': self.backup_queries,
            'memory_bytes': self.get_memory_usage(),
            'memory_kb': self.get_memory_usage() / 1024,
            'positive_count': len(self.positive_set),
            'n_regions': self.n_regions,
        }
