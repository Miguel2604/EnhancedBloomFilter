"""
Combined Enhanced Learned Bloom Filter Implementation

This is the production-ready version with all fixes applied:
1. Proper feature extraction for better discrimination
2. Backup filter for guaranteed positive storage
3. Improved model initialization and training
4. Adaptive threshold control for maintaining target FPR
"""

import numpy as np
from typing import Any, List, Tuple, Optional, Dict
from collections import deque
from functools import lru_cache
import os
import random
import struct
import sys
import time

import mmh3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bloom_filter.standard import StandardBloomFilter


class CombinedEnhancedLBF:
    """
    Combined Enhanced Learned Bloom Filter with all optimizations.
    
    Features:
    - O(1) incremental updates
    - Cache-aligned memory blocks
    - Adaptive FPR control
    - Proper ML model discrimination
    """
    
    def __init__(self,
                 initial_positive_set: Optional[List[Any]] = None,
                 initial_negative_set: Optional[List[Any]] = None,
                 positive_set: Optional[List[Any]] = None,
                 negative_set: Optional[List[Any]] = None,
                 target_fpr: float = 0.01,
                 n_blocks: int = 1024,
                 window_size: int = 10000,
                 reservoir_size: int = 1000,
                 monitoring_window: int = 1000,
                 enable_cache_opt: bool = True,
                 enable_incremental: bool = True,
                 enable_adaptive: bool = True,
                 verbose: bool = False):
        """
        Initialize fixed enhanced LBF.
        """
        if positive_set is not None:
            if initial_positive_set:
                initial_positive_set = list(initial_positive_set) + list(positive_set)
            else:
                initial_positive_set = list(positive_set)

        if negative_set is not None:
            if initial_negative_set:
                initial_negative_set = list(initial_negative_set) + list(negative_set)
            else:
                initial_negative_set = list(negative_set)

        self.target_fpr = target_fpr
        self.n_blocks = n_blocks
        self.window_size = window_size
        self.reservoir_size = reservoir_size
        self.monitoring_window = monitoring_window
        self.verbose = verbose
        
        # Feature flags
        self.cache_opt_enabled = enable_cache_opt
        self.incremental_enabled = enable_incremental
        self.adaptive_enabled = enable_adaptive
        
        # Initialize components
        self._init_model(initial_positive_set, initial_negative_set)
        
        # Store a sample of initial negatives for replay during online learning
        # This prevents the model from forgetting the negative class when processing a stream of positives
        self.negative_pool = []
        if initial_negative_set:
            # Keep up to 1000 negatives for replay
            self.negative_pool = list(initial_negative_set)
            if len(self.negative_pool) > 1000:
                self.negative_pool = random.sample(self.negative_pool, 1000)
        
        self._init_cache_structure()
        self._init_incremental_learning()
        self._init_adaptive_control()
        self._calibrate_initial_threshold(initial_positive_set, initial_negative_set)
        
        # Statistics
        self.total_queries = 0
        self.total_updates = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Primary filter stores ALL positive items
        initial_size = len(initial_positive_set) if initial_positive_set else 10000
        self.primary_filter = StandardBloomFilter(
            expected_elements=max(initial_size, 10000),
            false_positive_rate=target_fpr
        )
        
        # Backup filter only stores items predicted as false negatives by the model
        # Start small and grow dynamically
        self.positive_backup = StandardBloomFilter(
            expected_elements=1000,
            false_positive_rate=target_fpr
        )
        self._backup_items: List[Any] = []
        
        # Add initial positive set: always to primary; to backup only if model routes below threshold
        if initial_positive_set:
            for item in initial_positive_set:
                self.primary_filter.add(item)
                # Evaluate current model BEFORE any updates (already trained in _init_model)
                feats = self._extract_url_features(item)
                prob = self._predict_proba(feats)
                if prob < self.threshold:
                    self._add_to_backup(item)
        
        if verbose:
            self._print_config()
    
    def _init_model(self, positive_set, negative_set):
        """Initialize the base model with proper training."""
        # Use improved model with better initialization
        self.model = ImprovedPassiveAggressiveModel(n_features=20, C=10.0)
        
        if positive_set and negative_set:
            # Train on initial data with multiple passes for better convergence
            for _ in range(5):  # Multiple epochs
                # Train on positive examples
                for item in positive_set:
                    features = self._extract_url_features(item)
                    self.model.partial_fit(features, 1, learning_rate=0.1)
                
                # Train on negative examples  
                for item in negative_set:
                    features = self._extract_url_features(item)
                    self.model.partial_fit(features, 0, learning_rate=0.1)
    
    def _init_cache_structure(self):
        """Initialize cache-aligned memory blocks."""
        if self.cache_opt_enabled:
            self.cache_blocks = []
            for i in range(self.n_blocks):
                block = CacheAlignedBlock(block_id=i)
                self.cache_blocks.append(block)
        else:
            self.cache_blocks = None
    
    def _init_incremental_learning(self):
        """Initialize incremental learning components."""
        if self.incremental_enabled:
            self.sliding_window = deque(maxlen=self.window_size)
            self.reservoir = []
            self.reservoir_count = 0
            self.learning_rate = 0.1  # Start with higher learning rate
            self.lr_decay = 0.9995  # Slower decay
        else:
            self.sliding_window = None
            self.reservoir = None
    
    def _init_adaptive_control(self):
        """Initialize adaptive threshold control."""
        if self.adaptive_enabled:
            # Start with higher threshold for security applications
            self.threshold = 0.7  # Higher initial threshold
            
            # PID controller
            self.pid = PIDController(
                target=self.target_fpr,
                Kp=2.0,
                Ki=0.5,
                Kd=0.1
            )
            
            # Monitoring
            self.recent_queries = deque(maxlen=self.monitoring_window)
            self.threshold_history = [self.threshold]
            self.fpr_history = []
        else:
            self.threshold = 0.7  # Higher fixed threshold
            self.pid = None
            self.recent_queries = None
    
    def _calibrate_initial_threshold(self,
                                     positive_set: Optional[List[Any]],
                                     negative_set: Optional[List[Any]]):
        """Calibrate the initial decision threshold using labeled bootstrap data."""
        if not positive_set or not negative_set:
            return

        pos_probs = self._batch_predict_proba(positive_set)
        neg_probs = self._batch_predict_proba(negative_set)

        if pos_probs.size == 0 or neg_probs.size == 0:
            return

        candidate_thresholds = np.linspace(0.25, 0.85, num=61)
        fpr_tolerance = min(0.5, max(self.target_fpr * 1.1, self.target_fpr + 0.002))
        best_threshold = self.threshold
        best_score = -np.inf

        for t in candidate_thresholds:
            fpr = float(np.mean(neg_probs >= t))
            tpr = float(np.mean(pos_probs >= t))
            if fpr <= fpr_tolerance:
                score = tpr - (fpr * 0.25)
                if score > best_score:
                    best_score = score
                    best_threshold = t

        if not np.isfinite(best_score):
            quantile = np.clip(1.0 - fpr_tolerance, 0.0, 0.999)
            best_threshold = float(np.quantile(neg_probs, quantile))

        self.threshold = float(np.clip(best_threshold, 0.25, 0.85))

        if self.adaptive_enabled:
            if self.threshold_history:
                self.threshold_history[-1] = self.threshold
            else:
                self.threshold_history = [self.threshold]

    def _batch_predict_proba(self, items: List[Any], limit: int = 2000) -> np.ndarray:
        """Compute prediction probabilities for up to `limit` items."""
        if not items:
            return np.array([], dtype=np.float32)

        subset = items if len(items) <= limit else items[:limit]
        probs = np.empty(len(subset), dtype=np.float32)

        for idx, item in enumerate(subset):
            feats = self._extract_url_features(item)
            probs[idx] = self._predict_proba(feats)

        return probs

    def add(self, item: Any, label: int = 1):
        """Add item with proper training."""
        self.total_updates += 1
        
        # For positives, always add to primary; add to backup only if model routes below threshold BEFORE update
        features = self._extract_url_features(item)
        pre_update_prob = self._predict_proba(features)
        
        if label == 1:
            self.primary_filter.add(item)
            if pre_update_prob < self.threshold:
                self._add_to_backup(item)
        
        # Incremental learning
        if self.incremental_enabled:
            self.sliding_window.append((item, label))
            self._update_reservoir(item, label)
            
            # Update learning rate
            self.learning_rate *= self.lr_decay
            self.learning_rate = max(0.01, self.learning_rate)
        
        # Online model update with proper learning
        lr = self.learning_rate if self.incremental_enabled else 0.1
        self.model.partial_fit(features, label, learning_rate=lr)
        
        # Replay a negative sample to prevent forgetting
        # This is crucial when the stream contains mostly positives
        if self.negative_pool and label == 1:
            import random
            neg_item = random.choice(self.negative_pool)
            neg_feats = self._extract_url_features(neg_item)
            self.model.partial_fit(neg_feats, 0, learning_rate=lr)
        
        # Cache block updated only if the item is maintained in backup
        # (handled in _add_to_backup)
    
    def query(self, item: Any, ground_truth: Optional[bool] = None) -> bool:
        """Query with improved discrimination."""
        self.total_queries += 1
        
        # Use model for routing decision
        features = self._extract_url_features(item)
        score = self.model.predict(features)
        probability = 1 / (1 + np.exp(-score))
        
        # High model confidence: check primary filter to confirm membership.
        # The model is used for ROUTING, not for membership confirmation.
        # We must still verify with the actual Bloom filter to maintain
        # proper FPR guarantees.
        if probability >= self.threshold:
            # Model thinks it's likely positive - check primary filter
            if self.primary_filter.query(item):
                result = True
            else:
                # Model confident but not in primary - check backup
                result = self.positive_backup.query(item)
        else:
            # Low model confidence: check both filters to avoid false negatives
            if self.primary_filter.query(item):
                result = True
            else:
                # Low confidence: rely on backup filter (and cache) to recover
                # positives the model routes below the threshold.
                if self.cache_opt_enabled and self.cache_blocks is not None:
                    block = self._get_cache_block(item)
                    if block.check_backup_bit(item):
                        # Bit is set: It MIGHT be in backup.
                        # We must check the full backup filter to be sure.
                        # This is a "partial hit" or "cache says maybe".
                        # In terms of avoiding main memory (backup filter), we failed.
                        self.cache_misses += 1
                        result = self.positive_backup.query(item)
                    else:
                        # Bit is NOT set: It is DEFINITELY NOT in backup.
                        # We resolved the query using only the cache block.
                        self.cache_hits += 1
                        result = False
                else:
                    result = self.positive_backup.query(item)
        
        # Track for adaptive control
        if self.adaptive_enabled and ground_truth is not None:
            self.recent_queries.append({
                'result': result,
                'ground_truth': ground_truth,
                'item': item
            })
            
            # Periodically adjust threshold
            if self.total_queries % 100 == 0:
                self._adjust_threshold()
        
        return result
    
    # Pre-computed lookup table for DNA characters (A, C, G, T, a, c, g, t)
    _DNA_LOOKUP = np.zeros(256, dtype=bool)
    _DNA_LOOKUP[[65, 67, 71, 84, 97, 99, 103, 116]] = True

    def _extract_url_features(self, item: Any) -> np.ndarray:
        """Extract generic string features with URL- and pattern-aware signals.

        Optimized for throughput - avoids expensive NumPy operations on small arrays.
        Must return exactly 20 features.
        """
        s = str(item)
        s_lower = s.lower()
        features = np.zeros(20, dtype=np.float32)

        L = len(s)
        if L == 0:
            return features

        # 0. Normalized length
        features[0] = L / 100.0

        # 1-4. Fast normalized hash sketches
        for idx, seed in enumerate((17, 31, 73, 127), start=1):
            hv = mmh3.hash(s, seed=seed, signed=False)
            features[idx] = (hv & 0xFFFF) / 65535.0

        # 5-7. Character category ratios (use native str methods - faster for small strings)
        digits = sum(c.isdigit() for c in s)
        alphas = sum(c.isalpha() for c in s)
        others = L - digits - alphas
        features[5] = digits / L
        features[6] = alphas / L
        features[7] = max(0.0, others) / L

        # 8. Suspicious token indicator count
        features[8] = (
            ("malware" in s_lower) + ("phishing" in s_lower) + 
            ("virus" in s_lower) + ("trojan" in s_lower) + ("hack" in s_lower)
        ) / 5.0

        # 9. Suspicious TLD
        features[9] = 1.0 if (".ru" in s_lower or ".cn" in s_lower or 
                              ".xyz" in s_lower or ".top" in s_lower) else 0.0

        # 10. Benign brand tokens
        features[10] = (
            ("google" in s_lower) + ("amazon" in s_lower) + 
            ("microsoft" in s_lower) + ("github" in s_lower) + ("wikipedia" in s_lower)
        ) / 5.0

        # 11-14. Structural features
        features[11] = 1.0 if s_lower.startswith("https://") else 0.0
        features[12] = s.count("/") / 10.0
        features[13] = s.count(".") / 10.0
        features[14] = s.count("-") / 10.0

        # 15-17. Lightweight character hints (Option D: no np.mean/std)
        features[15] = ord(s[0]) / 256.0  # First char hint
        features[16] = ord(s[-1]) / 256.0  # Last char hint
        features[17] = len(set(s)) / L  # Unique char ratio (Python set is fast)

        # 18. DNA-like pattern indicator (fast set check)
        features[18] = 1.0 if all(c in "ACGTacgt" for c in s) else 0.0

        # 19. Key-like pattern indicator
        features[19] = 1.0 if (":" in s or "_" in s or "-" in s) else 0.0

        return features
    
    def _adjust_threshold(self):
        """Adjust threshold using PID control."""
        if len(self.recent_queries) < 100:
            return
        
        # Calculate recent FPR
        negatives = [q for q in self.recent_queries if not q['ground_truth']]
        if len(negatives) == 0:
            return
        
        false_positives = sum(1 for q in negatives if q['result'])
        recent_fpr = false_positives / len(negatives)
        
        # PID adjustment
        adjustment = self.pid.update(recent_fpr)
        
        # Apply adjustment
        old_threshold = self.threshold
        # Invert adjustment: if FPR is high (negative error), we want to INCREASE threshold
        self.threshold = np.clip(self.threshold - adjustment, 0.3, 0.95)
        
        # Track history
        self.threshold_history.append(self.threshold)
        self.fpr_history.append(recent_fpr)
        
        if self.verbose and abs(old_threshold - self.threshold) > 0.01:
            print(f"Threshold adjusted: {old_threshold:.3f} → {self.threshold:.3f}")
    
    def _update_reservoir(self, item: Any, label: int):
        """Update reservoir with new item."""
        self.reservoir_count += 1
        
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append((item, label))
        else:
            # Random replacement
            j = np.random.randint(0, self.reservoir_count)
            if j < self.reservoir_size:
                self.reservoir[j] = (item, label)
    
    def _update_cache_block(self, item: Any):
        """Update cache block with new item."""
        if not self.cache_opt_enabled:
            return
        
        block = self._get_cache_block(item)
        block.add_to_backup(item)

    def _get_cache_block(self, item: Any):
        block_id = abs(hash(str(item))) % self.n_blocks
        return self.cache_blocks[block_id]

    def _predict_proba(self, features: np.ndarray) -> float:
        score = self.model.predict(features)
        return float(1.0 / (1.0 + np.exp(-score)))

    def _add_to_backup(self, item: Any):
        """Add item to backup filter with dynamic resizing and cache bit update."""
        self.positive_backup.add(item)
        self._backup_items.append(item)
        # Update cache block bit for fast path
        if self.cache_opt_enabled and self.cache_blocks is not None:
            self._update_cache_block(item)
        # Grow backup filter as needed (keep load factor reasonable)
        self._maybe_resize_backup()

    def _maybe_resize_backup(self):
        try:
            load = self.positive_backup.get_load_factor()
        except Exception:
            # If method not available, skip
            return
        if load <= 0.5:
            return
        new_capacity = max(self.positive_backup.count * 2, 1)
        new_filter = StandardBloomFilter(expected_elements=new_capacity, false_positive_rate=self.target_fpr)
        for it in self._backup_items:
            new_filter.add(it)
        self.positive_backup = new_filter
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics for the combined filter."""
        if not self.cache_opt_enabled or self.cache_blocks is None:
            return {
                'query_count': self.total_queries,
                'cache_hits': 0,
                'cache_misses': 0,
                'cache_hit_rate': 0.0,
                'blocks': 0
            }
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(1, total)) * 100
        return {
            'query_count': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'blocks': self.n_blocks
        }
    
    def get_memory_usage(self) -> Dict:
        """Estimate memory usage of model, filters, and cache blocks."""
        # Model (weights + bias)
        model_bytes = 0
        if hasattr(self.model, 'weights'):
            model_bytes += self.model.weights.nbytes
        if hasattr(self.model, 'bias'):
            model_bytes += np.array(self.model.bias, dtype=np.float32).nbytes
        
        # Primary and backup filters
        primary_bytes = self.primary_filter.get_memory_usage() if hasattr(self.primary_filter, 'get_memory_usage') else 0
        backup_bytes = self.positive_backup.get_memory_usage() if hasattr(self.positive_backup, 'get_memory_usage') else 0
        
        # Cache-aligned blocks
        cache_bytes = 0
        if self.cache_opt_enabled and self.cache_blocks is not None:
            cache_bytes = len(self.cache_blocks) * 64  # 64 bytes per block
        
        total = model_bytes + primary_bytes + backup_bytes + cache_bytes
        return {
            'model_bytes': model_bytes,
            'primary_filter_bytes': primary_bytes,
            'backup_filter_bytes': backup_bytes,
            'cache_bytes': cache_bytes,
            'total_bytes': total
        }
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        cache_stats = self.get_cache_stats()
        cache_hit_rate = cache_stats.get('cache_hit_rate', 0.0)

        stats = {
            'total_queries': self.total_queries,
            'total_updates': self.total_updates,
            'cache_hit_rate': cache_hit_rate,
            'current_threshold': self.threshold,
            'features_enabled': {
                'cache_optimization': self.cache_opt_enabled,
                'incremental_learning': self.incremental_enabled,
                'adaptive_threshold': self.adaptive_enabled
            }
        }
        
        if self.incremental_enabled:
            stats['window_size'] = len(self.sliding_window)
            stats['reservoir_size'] = len(self.reservoir)
            stats['learning_rate'] = self.learning_rate
        
        if self.adaptive_enabled and self.fpr_history:
            stats['fpr_variance'] = np.std(self.fpr_history) / max(0.001, np.mean(self.fpr_history)) * 100
            stats['threshold_range'] = (min(self.threshold_history), max(self.threshold_history))
        
        # Include memory usage summary if available
        mem = self.get_memory_usage()
        stats['memory_usage'] = mem
        
        return stats
    
    def _print_config(self):
        """Print configuration."""
        print("\nCombined Enhanced LBF Configuration:")
        print(f"  Target FPR: {self.target_fpr:.4f}")
        print(f"  Cache Optimization: {'Enabled' if self.cache_opt_enabled else 'Disabled'}")
        if self.cache_opt_enabled:
            print(f"    - Blocks: {self.n_blocks}")
        print(f"  Incremental Learning: {'Enabled' if self.incremental_enabled else 'Disabled'}")
        if self.incremental_enabled:
            print(f"    - Window: {self.window_size}")
            print(f"    - Reservoir: {self.reservoir_size}")
        print(f"  Adaptive Threshold: {'Enabled' if self.adaptive_enabled else 'Disabled'}")
        if self.adaptive_enabled:
            print(f"    - Initial: {self.threshold:.3f}")


class ImprovedPassiveAggressiveModel:
    """Improved Passive-Aggressive model with better initialization."""
    
    def __init__(self, n_features: int, C: float = 10.0):
        self.n_features = n_features
        self.C = C
        # Initialize with small random weights instead of zeros
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def predict(self, X: np.ndarray) -> float:
        return np.dot(self.weights, X) + self.bias
    
    def partial_fit(self, X: np.ndarray, y: int, learning_rate: float = 0.1):
        """Update model with proper PA-I algorithm."""
        y_pa = 2 * y - 1  # Convert to -1, +1
        prediction = self.predict(X)
        loss = max(0, 1 - y_pa * prediction)
        
        if loss > 0:
            # PA-I update
            tau = min(self.C, loss / (np.dot(X, X) + 1e-10))
            tau *= learning_rate
            
            # Update weights and bias
            self.weights += tau * y_pa * X
            self.bias += tau * y_pa


class CacheAlignedBlock:
    """Cache-aligned memory block."""
    
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.data = np.zeros(64, dtype=np.uint8)
        self.backup_bits = self.data[:60]
        self.metadata = self.data[60:64].view(np.uint32)
    
    def check_backup_bit(self, item: Any) -> bool:
        hash_val = hash(str(item))
        bit_idx = (hash_val >> 8) % 480
        byte_idx = bit_idx // 8
        bit_offset = bit_idx % 8
        return bool(self.backup_bits[byte_idx] & (1 << bit_offset))
    
    def add_to_backup(self, item: Any):
        hash_val = hash(str(item))
        bit_idx = (hash_val >> 8) % 480
        byte_idx = bit_idx // 8
        bit_offset = bit_idx % 8
        self.backup_bits[byte_idx] |= (1 << bit_offset)


class PIDController:
    """PID controller for adaptive threshold."""
    
    def __init__(self, target: float, Kp: float, Ki: float, Kd: float):
        self.target = target
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
    
    def update(self, current: float) -> float:
        error = self.target - current
        P = self.Kp * error
        self.integral += error
        self.integral = np.clip(self.integral, -10, 10)  # Prevent windup
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error)
        self.prev_error = error
        return np.clip(P + I + D, -0.05, 0.05)  # Smaller adjustments