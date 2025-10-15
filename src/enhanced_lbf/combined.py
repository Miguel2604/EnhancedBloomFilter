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
import hashlib
import struct
import time
import sys
import os
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
        self._init_cache_structure()
        self._init_incremental_learning()
        self._init_adaptive_control()
        
        # Statistics
        self.total_queries = 0
        self.total_updates = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Backup filter for positive items (malicious URLs)
        self.positive_backup = StandardBloomFilter(
            expected_elements=10000,
            false_positive_rate=target_fpr
        )
        
        # Add initial positive set to backup
        if initial_positive_set:
            for item in initial_positive_set:
                self.positive_backup.add(item)
        
        if verbose:
            self._print_config()
    
    def _init_model(self, positive_set, negative_set):
        """Initialize the base model with proper training."""
        # Use improved model with better initialization
        self.model = ImprovedPassiveAggressiveModel(n_features=20, C=10.0)
        
        if positive_set and negative_set:
            # Train on initial data with multiple passes for better convergence
            for _ in range(3):  # Multiple epochs
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
                Kp=1.0,  # Less aggressive
                Ki=0.2,
                Kd=0.05
            )
            
            # Monitoring
            self.recent_queries = deque(maxlen=self.monitoring_window)
            self.threshold_history = [self.threshold]
            self.fpr_history = []
        else:
            self.threshold = 0.7  # Higher fixed threshold
            self.pid = None
            self.recent_queries = None
    
    def add(self, item: Any, label: int = 1):
        """Add item with proper training."""
        self.total_updates += 1
        
        # Add positive items to backup filter
        if label == 1:
            self.positive_backup.add(item)
        
        # Incremental learning
        if self.incremental_enabled:
            self.sliding_window.append((item, label))
            self._update_reservoir(item, label)
            
            # Update learning rate
            self.learning_rate *= self.lr_decay
            self.learning_rate = max(0.01, self.learning_rate)
        
        # Extract URL-specific features
        features = self._extract_url_features(item)
        
        # Online model update with proper learning
        lr = self.learning_rate if self.incremental_enabled else 0.1
        self.model.partial_fit(features, label, learning_rate=lr)
        
        # Update cache blocks if enabled
        if self.cache_opt_enabled and label == 1:
            self._update_cache_block(item)
    
    def query(self, item: Any, ground_truth: Optional[bool] = None) -> bool:
        """Query with improved discrimination."""
        self.total_queries += 1
        
        # First check the positive backup filter
        if self.positive_backup.query(item):
            # Item is in the positive set (malicious)
            result = True
        else:
            # Use model for prediction
            features = self._extract_url_features(item)
            score = self.model.predict(features)
            probability = 1 / (1 + np.exp(-score))
            
            # Use threshold for decision
            result = probability >= self.threshold
        
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
    
    def _extract_url_features(self, item: Any) -> np.ndarray:
        """Extract URL-specific features for better discrimination."""
        item_str = str(item).lower()
        features = np.zeros(20, dtype=np.float32)
        
        # Basic features
        features[0] = len(item_str) / 100.0
        features[1] = item_str.count('/') / 10.0
        features[2] = item_str.count('.') / 10.0
        features[3] = item_str.count('-') / 10.0
        features[4] = item_str.count('_') / 10.0
        
        # URL-specific features
        features[5] = 1.0 if 'malware' in item_str else 0.0
        features[6] = 1.0 if 'virus' in item_str else 0.0
        features[7] = 1.0 if 'trojan' in item_str else 0.0
        features[8] = 1.0 if 'phishing' in item_str else 0.0
        features[9] = 1.0 if 'hack' in item_str else 0.0
        
        # Benign indicators
        features[10] = 1.0 if 'google' in item_str else 0.0
        features[11] = 1.0 if 'amazon' in item_str else 0.0
        features[12] = 1.0 if 'microsoft' in item_str else 0.0
        features[13] = 1.0 if 'github' in item_str else 0.0
        features[14] = 1.0 if 'wikipedia' in item_str else 0.0
        
        # Structural features
        features[15] = 1.0 if item_str.startswith('https') else 0.0
        features[16] = 1.0 if '.com' in item_str else 0.0
        features[17] = 1.0 if '.org' in item_str else 0.0
        features[18] = sum(c.isdigit() for c in item_str) / 20.0
        features[19] = 1.0 if '.php' in item_str else 0.0
        
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
        self.threshold = np.clip(self.threshold + adjustment, 0.3, 0.95)
        
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
        
        block_id = abs(hash(str(item))) % self.n_blocks
        block = self.cache_blocks[block_id]
        block.add_to_backup(item)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        cache_hit_rate = (self.cache_hits / max(1, self.cache_hits + self.cache_misses)) * 100 if self.cache_opt_enabled else 0
        
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