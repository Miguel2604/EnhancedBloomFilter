"""
Partitioned Learned Bloom Filter (PLBF) Implementation

Reference: Vaidya et al. (2020) - "Partitioned Learned Bloom Filter"

Key Innovation: Frames model utilization as an optimization problem and uses
dynamic programming to find near-optimal partitioning of the key space.
"""

import numpy as np
from typing import Any, List, Optional, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bloom_filter.standard import StandardBloomFilter


class PartitionedLearnedBloomFilter:
    """
    Partitioned Learned Bloom Filter with DP-based optimization.
    
    The key idea is to partition positive items based on model confidence scores,
    then use dynamic programming to optimally allocate backup filter sizes.
    """
    
    def __init__(self,
                 positive_set: List[Any],
                 negative_set: List[Any],
                 target_fpr: float = 0.01,
                 n_partitions: int = 8,
                 memory_budget: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize PLBF.
        
        Args:
            positive_set: Training positive examples
            negative_set: Training negative examples
            target_fpr: Target false positive rate
            n_partitions: Number of partitions
            memory_budget: Optional memory constraint in bytes
            verbose: Print initialization details
        """
        self.target_fpr = target_fpr
        self.n_partitions = n_partitions
        self.memory_budget = memory_budget
        self.verbose = verbose
        
        if not positive_set or not negative_set:
            raise ValueError("Both positive and negative training sets required")
        
        # Store sets
        self.positive_set = set(positive_set)
        self.negative_set = set(negative_set)
        
        # Train model
        self._train_model(positive_set, negative_set)
        
        # Partition items and optimize backup filter allocation
        self._optimize_partitions(positive_set)
        
        # Statistics
        self.total_queries = 0
        self.model_predictions = 0
        self.backup_queries = 0
        
        if verbose:
            print(f"Initialized PLBF:")
            print(f"  Positive training examples: {len(positive_set):,}")
            print(f"  Negative training examples: {len(negative_set):,}")
            print(f"  Target FPR: {target_fpr:.4f}")
            print(f"  Number of partitions: {n_partitions}")
            print(f"  Model memory: ~{self._estimate_model_size() / 1024:.2f} KB")
            print(f"  Total memory: ~{self.get_memory_usage() / 1024:.2f} KB")
    
    def _extract_features(self, item: Any) -> np.ndarray:
        """Extract features from an item."""
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
        
        # Hash-based features
        import hashlib
        hash_bytes = hashlib.sha256(item_str.encode()).digest()
        for i in range(0, 12, 2):
            features.append(int.from_bytes(hash_bytes[i:i+2], 'big'))
        
        return np.array(features, dtype=np.float32)
    
    def _train_model(self, positive_set: List[Any], negative_set: List[Any]):
        """Train the ML classifier."""
        X_pos = np.array([self._extract_features(item) for item in positive_set])
        X_neg = np.array([self._extract_features(item) for item in negative_set])
        
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(positive_set) + [0] * len(negative_set))
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier with probability estimates
        self.model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            random_state=42,
            warm_start=True
        )
        
        # Train with multiple epochs
        for _ in range(3):
            indices = np.random.permutation(len(X))
            self.model.partial_fit(X_scaled[indices], y[indices], classes=[0, 1])
    
    def _optimize_partitions(self, positive_set: List[Any]):
        """
        Use DP to optimize partition allocation.
        
        This is a simplified version of the optimization in the paper.
        The full version would solve a more complex optimization problem.
        """
        # Predict scores for all positive items
        X_pos = np.array([self._extract_features(item) for item in positive_set])
        X_pos_scaled = self.scaler.transform(X_pos)
        scores = self.model.decision_function(X_pos_scaled)
        
        # Sort items by confidence score (higher = more confident positive)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_items = [positive_set[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        # Partition items into n_partitions groups
        partition_size = len(sorted_items) // self.n_partitions
        self.partitions = []
        self.partition_scores = []
        
        for i in range(self.n_partitions):
            start = i * partition_size
            end = start + partition_size if i < self.n_partitions - 1 else len(sorted_items)
            self.partitions.append(sorted_items[start:end])
            if end > start:
                self.partition_scores.append(np.mean(sorted_scores[start:end]))
            else:
                self.partition_scores.append(0.0)
        
        # Allocate backup filter sizes using DP-inspired heuristic
        # Higher confidence partitions get smaller filters
        self._allocate_backup_filters()
    
    def _allocate_backup_filters(self):
        """
        Allocate backup filter sizes based on partition confidence.
        
        This uses a heuristic: higher confidence partitions need smaller backups.
        """
        self.backup_filters = []
        
        # Normalize scores to [0, 1] for allocation
        if len(self.partition_scores) > 0:
            min_score = min(self.partition_scores)
            max_score = max(self.partition_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for partition_items, score in zip(self.partitions, self.partition_scores):
                if not partition_items:
                    bf = StandardBloomFilter(expected_elements=10, false_positive_rate=self.target_fpr)
                else:
                    # Higher score = more confident = smaller FPR needed
                    normalized_score = (score - min_score) / score_range
                    # Scale FPR: high confidence gets target FPR, low confidence gets higher FPR
                    partition_fpr = self.target_fpr * (1 + (1 - normalized_score))
                    partition_fpr = min(partition_fpr, 0.1)  # Cap at 10%
                    
                    bf = StandardBloomFilter(
                        expected_elements=len(partition_items),
                        false_positive_rate=partition_fpr
                    )
                    for item in partition_items:
                        bf.add(item)
                
                self.backup_filters.append(bf)
        else:
            # Fallback: single partition
            bf = StandardBloomFilter(
                expected_elements=max(len(self.positive_set), 10),
                false_positive_rate=self.target_fpr
            )
            for item in self.positive_set:
                bf.add(item)
            self.backup_filters.append(bf)
    
    def _get_partition_index(self, item: Any) -> int:
        """Determine which partition an item belongs to based on its score."""
        # Check if item already in a partition (for training items)
        for i, partition in enumerate(self.partitions):
            if item in partition:
                return i
        
        # For new items, use model score
        features = self._extract_features(item)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        score = self.model.decision_function(features_scaled)[0]
        
        # Find closest partition by score
        # Higher scores -> earlier partitions (more confident positives)
        if not self.partition_scores:
            return 0
        
        # Find partition with closest mean score
        min_diff = float('inf')
        best_idx = 0
        for i, threshold_score in enumerate(self.partition_scores):
            diff = abs(score - threshold_score)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        
        return best_idx
    
    def add(self, item: Any):
        """Add an item to the appropriate partition."""
        partition_idx = self._get_partition_index(item)
        if 0 <= partition_idx < len(self.backup_filters):
            self.backup_filters[partition_idx].add(item)
        self.positive_set.add(item)
    
    def query(self, item: Any) -> bool:
        """Query whether an item is in the set."""
        self.total_queries += 1
        self.model_predictions += 1
        
        # Determine partition
        partition_idx = self._get_partition_index(item)
        
        # Query appropriate backup filter
        self.backup_queries += 1
        if 0 <= partition_idx < len(self.backup_filters):
            return self.backup_filters[partition_idx].query(item)
        return False
    
    def _estimate_model_size(self) -> int:
        """Estimate memory usage of the model in bytes."""
        model_size = 0
        if hasattr(self.model, 'coef_'):
            model_size += self.model.coef_.nbytes
        if hasattr(self.model, 'intercept_'):
            model_size += self.model.intercept_.nbytes
        model_size += self.scaler.mean_.nbytes + self.scaler.scale_.nbytes
        return model_size
    
    def get_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        total = self._estimate_model_size()
        for bf in self.backup_filters:
            total += bf.m // 8
        return total
    
    def get_stats(self) -> dict:
        """Get statistics about filter performance."""
        return {
            'total_queries': self.total_queries,
            'model_predictions': self.model_predictions,
            'backup_queries': self.backup_queries,
            'memory_bytes': self.get_memory_usage(),
            'memory_kb': self.get_memory_usage() / 1024,
            'positive_count': len(self.positive_set),
            'n_partitions': self.n_partitions,
            'partition_sizes': [len(p) for p in self.partitions],
        }
