"""
Stable Learned Bloom Filter (s-SLBF) Implementation

Reference: Liu et al. (2020) - "Stable Learned Bloom Filters for Data Streams" (VLDB)

Key Innovation: Designed for dynamic data streams with frequent insertions.
Maintains constant expected FPR despite continuous member updates through
updatable backup filters.
"""

import numpy as np
from typing import Any, List, Optional
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bloom_filter.standard import StandardBloomFilter


class StableLearnedBloomFilter:
    """
    Stable Learned Bloom Filter for data streams (s-SLBF variant).
    
    Key features:
    - Maintains stable FPR under continuous insertions
    - Uses updatable backup filters that can grow dynamically
    - Periodically retrains model on recent data
    """
    
    def __init__(self,
                 positive_set: List[Any],
                 negative_set: List[Any],
                 target_fpr: float = 0.01,
                 retrain_threshold: int = 1000,
                 buffer_size: int = 10000,
                 retain_full_history: bool = False,
                 verbose: bool = False):
        """
        Initialize Stable LBF.
        
        Args:
            positive_set: Initial positive training examples
            negative_set: Initial negative training examples
            target_fpr: Target false positive rate
            retrain_threshold: Number of insertions before retraining
            buffer_size: Size of recent items buffer for retraining
            verbose: Print initialization details
        """
        self.target_fpr = target_fpr
        self.retrain_threshold = retrain_threshold
        self.buffer_size = buffer_size
        self.retain_full_history = retain_full_history
        self.verbose = verbose
        
        if not positive_set or not negative_set:
            raise ValueError("Both positive and negative training sets required")
        
        # Store sets
        self.positive_set = set(positive_set)
        self.negative_set = set(negative_set)
        
        # Buffer for recent insertions
        self.recent_positives = deque(maxlen=buffer_size)
        self.recent_negatives = deque(maxlen=buffer_size // 10)  # Smaller negative buffer
        
        # Initialize with training data
        for item in positive_set:
            self.recent_positives.append(item)
        for item in negative_set[:buffer_size // 10]:
            self.recent_negatives.append(item)
        
        # Train initial model
        self._train_model(list(self.recent_positives), list(self.recent_negatives))
        
        # Initialize backup filters
        self._init_backup_filters(positive_set)
        
        # Counters
        self.insertions_since_retrain = 0
        self.total_queries = 0
        self.total_insertions = 0
        self.model_rejections = 0
        self.backup_queries = 0
        self.retrain_count = 0
        
        if verbose:
            print(f"Initialized Stable LBF:")
            print(f"  Positive training examples: {len(positive_set):,}")
            print(f"  Negative training examples: {len(negative_set):,}")
            print(f"  Target FPR: {target_fpr:.4f}")
            print(f"  Retrain threshold: {retrain_threshold:,}")
            print(f"  Buffer size: {buffer_size:,}")
            if retain_full_history:
                print("  Backup retention: full history")
    
    def _extract_features(self, item: Any) -> np.ndarray:
        """Extract features from an item."""
        item_str = str(item)
        
        features = []
        features.append(len(item_str))
        features.append(item_str.count('.'))
        features.append(item_str.count('/'))
        features.append(item_str.count('-'))
        features.append(item_str.count('_'))
        features.append(sum(c.isdigit() for c in item_str))
        features.append(sum(c.isupper() for c in item_str))
        features.append(sum(c.islower() for c in item_str))
        
        import hashlib
        hash_bytes = hashlib.sha256(item_str.encode()).digest()
        for i in range(0, 12, 2):
            features.append(int.from_bytes(hash_bytes[i:i+2], 'big'))
        
        return np.array(features, dtype=np.float32)
    
    def _train_model(self, positive_samples: List[Any], negative_samples: List[Any]):
        """Train or retrain the ML classifier."""
        if not positive_samples or not negative_samples:
            return
        
        X_pos = np.array([self._extract_features(item) for item in positive_samples])
        X_neg = np.array([self._extract_features(item) for item in negative_samples])
        
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(positive_samples) + [0] * len(negative_samples))
        
        # Normalize features
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Update scaler with new data
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
        
        # Train or update classifier
        if not hasattr(self, 'model'):
            self.model = SGDClassifier(
                loss='hinge',  # SVM-like for better margins
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                random_state=42,
                warm_start=True
            )
            self.model.fit(X_scaled, y)
        else:
            # Incremental update with warm start
            for _ in range(3):
                indices = np.random.permutation(len(X))
                self.model.partial_fit(X_scaled[indices], y[indices])
    
    def _init_backup_filters(self, positive_set: List[Any]):
        """
        Initialize backup filters.
        
        For s-SLBF, we use a single backup filter that gets rebuilt periodically.
        According to the paper, the key is to rebuild the backup to prevent FPR degradation.
        """
        # Single backup filter - will be rebuilt during retraining
        # Size it for initial training set + some growth room
        initial_source = self.positive_set if self.retain_full_history else positive_set
        expected_size = max(len(initial_source), 100)
        
        self.backup_filter = StandardBloomFilter(
            expected_elements=expected_size,
            false_positive_rate=self.target_fpr
        )
        for item in initial_source:
            self.backup_filter.add(item)
    
    def add(self, item: Any):
        """
        Add an item to the filter.
        
        This is the key operation for streaming - must maintain stability.
        Key insight: We rebuild the backup filter during retraining to prevent FPR degradation.
        """
        self.total_insertions += 1
        self.insertions_since_retrain += 1
        
        # Add to backup filter (temporary until next rebuild)
        self.backup_filter.add(item)
        
        # Update positive set and buffer
        self.positive_set.add(item)
        self.recent_positives.append(item)
        
        # Check if we need to retrain
        if self.insertions_since_retrain >= self.retrain_threshold:
            self._retrain()
    
    def _retrain(self):
        """
        Retrain the model with recent data and rebuild backup filter.
        
        This is THE KEY to stability - we rebuild the backup filter to prevent FPR degradation.
        """
        if self.verbose:
            print(f"Retraining model (retrain #{self.retrain_count + 1})...")
        
        # Retrain with recent data
        positive_samples = list(self.recent_positives)
        negative_samples = list(self.recent_negatives) if self.recent_negatives else []
        
        # Generate synthetic negatives if we don't have enough
        if len(negative_samples) < len(positive_samples) // 10:
            import random
            import string
            for _ in range(len(positive_samples) // 10):
                random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
                if random_str not in self.positive_set:
                    negative_samples.append(random_str)
                    self.recent_negatives.append(random_str)
        
        self._train_model(positive_samples, negative_samples)
        
        # CRITICAL: Rebuild backup filter from scratch.
        # Optionally retain the full positive history instead of the sliding window.
        backup_source = self.positive_set if self.retain_full_history else list(self.recent_positives)
        expected_size = len(backup_source)
        self.backup_filter = StandardBloomFilter(
            expected_elements=max(expected_size, self.retrain_threshold),
            false_positive_rate=self.target_fpr
        )
        
        # Add positives to fresh backup
        for item in backup_source:
            self.backup_filter.add(item)
        
        # Reset counter
        self.insertions_since_retrain = 0
        self.retrain_count += 1
    
    def query(self, item: Any) -> bool:
        """
        Query whether an item is in the set.
        
        Returns True if likely positive, False if definitely negative.
        """
        self.total_queries += 1
        
        # For Stable LBF, we prioritize backup filter to avoid false negatives
        # This is key for stream stability
        
        # Check backup filter first
        if self.backup_filter.query(item):
            return True
        
        # If not in backup, use model as final check
        features = self._extract_features(item)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        if prediction == 0:
            self.model_rejections += 1
            return False
        
        # Model says positive but not in backup - could be new or FP
        self.backup_queries += 1
        return False  # Conservative: if not in backup, reject
    
    def _estimate_model_size(self) -> int:
        """Estimate memory usage of the model in bytes."""
        model_size = 0
        if hasattr(self.model, 'coef_'):
            model_size += self.model.coef_.nbytes
        if hasattr(self.model, 'intercept_'):
            model_size += self.model.intercept_.nbytes
        if hasattr(self, 'scaler'):
            model_size += self.scaler.mean_.nbytes + self.scaler.scale_.nbytes
        return model_size
    
    def get_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        total = self._estimate_model_size()
        total += self.backup_filter.m // 8
        return total
    
    def get_stats(self) -> dict:
        """Get statistics about filter performance."""
        return {
            'total_queries': self.total_queries,
            'total_insertions': self.total_insertions,
            'model_rejections': self.model_rejections,
            'backup_queries': self.backup_queries,
            'retrain_count': self.retrain_count,
            'insertions_since_retrain': self.insertions_since_retrain,
            'memory_bytes': self.get_memory_usage(),
            'memory_kb': self.get_memory_usage() / 1024,
            'positive_count': len(self.positive_set),
        }
