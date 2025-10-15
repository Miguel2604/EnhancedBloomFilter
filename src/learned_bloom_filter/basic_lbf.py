"""
Basic Learned Bloom Filter Implementation

The Learned Bloom Filter uses machine learning to replace hash functions,
achieving better space efficiency while maintaining the same FPR guarantees.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Any, List, Tuple, Optional
import hashlib
import pickle
from bitarray import bitarray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bloom_filter.standard import StandardBloomFilter


class BasicLearnedBloomFilter:
    """
    A Learned Bloom Filter that uses ML to reduce memory usage.
    
    Components:
    1. Machine Learning Model (predicts membership)
    2. Backup Bloom Filter (handles false negatives)
    
    Attributes:
        model: ML model for membership prediction
        backup_filter: Standard BF for false negatives
        threshold: Decision threshold for ML predictions
        scaler: Feature scaler for ML model
    """
    
    def __init__(self, 
                 positive_set: List[Any],
                 negative_set: List[Any],
                 target_fpr: float = 0.01,
                 threshold: float = 0.5,
                 model_type: str = 'logistic',
                 verbose: bool = False):
        """
        Initialize a Learned Bloom Filter.
        
        Args:
            positive_set: Training set of items that should be in the filter
            negative_set: Training set of items that should not be in the filter
            target_fpr: Target false positive rate
            threshold: Decision threshold for ML model (default 0.5)
            model_type: Type of ML model ('logistic', 'svm', 'tree')
            verbose: Print training progress
        """
        if len(positive_set) == 0:
            raise ValueError("Positive set cannot be empty")
        if len(negative_set) == 0:
            raise ValueError("Negative set cannot be empty")
            
        self.positive_set = positive_set
        self.negative_set = negative_set
        self.target_fpr = target_fpr
        self.threshold = threshold
        self.model_type = model_type
        self.verbose = verbose
        
        # Feature extraction setup
        self.scaler = StandardScaler()
        
        # Train the ML model
        self._train_model()
        
        # Find false negatives and create backup filter
        self._create_backup_filter()
        
        # Statistics
        self.query_count = 0
        self.positive_predictions = 0
        
        if verbose:
            self._print_statistics()
    
    def _extract_features(self, item: Any) -> np.ndarray:
        """
        Extract numerical features from an item.
        
        Args:
            item: Item to extract features from
            
        Returns:
            Feature vector as numpy array
        """
        # Convert to string for consistent hashing
        item_str = str(item)
        
        # Generate multiple hash values as features
        features = []
        
        # Feature 1: Length
        features.append(len(item_str))
        
        # Features 2-5: Hash values from different functions
        hash_funcs = [hashlib.md5, hashlib.sha1, hashlib.sha256, hashlib.sha512]
        for hash_func in hash_funcs:
            hash_val = int(hash_func(item_str.encode()).hexdigest()[:8], 16)
            features.append(hash_val)
        
        # Features 6-10: Character statistics
        features.append(sum(c.isdigit() for c in item_str))  # Digit count
        features.append(sum(c.isalpha() for c in item_str))  # Letter count
        features.append(sum(c.isspace() for c in item_str))  # Space count
        features.append(ord(item_str[0]) if item_str else 0)  # First char
        features.append(ord(item_str[-1]) if item_str else 0)  # Last char
        
        # Features 11-15: Statistical properties
        char_codes = [ord(c) for c in item_str]
        features.append(np.mean(char_codes) if char_codes else 0)
        features.append(np.std(char_codes) if char_codes else 0)
        features.append(min(char_codes) if char_codes else 0)
        features.append(max(char_codes) if char_codes else 0)
        features.append(len(set(item_str)))  # Unique chars
        
        return np.array(features, dtype=np.float32)
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from positive and negative sets.
        
        Returns:
            X: Feature matrix
            y: Label vector (1 for positive, 0 for negative)
        """
        # Extract features for all items
        positive_features = [self._extract_features(item) 
                            for item in self.positive_set]
        negative_features = [self._extract_features(item) 
                            for item in self.negative_set]
        
        # Combine into training data
        X = np.vstack(positive_features + negative_features)
        y = np.array([1] * len(positive_features) + 
                     [0] * len(negative_features))
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def _train_model(self):
        """Train the machine learning model."""
        if self.verbose:
            print(f"Training {self.model_type} model...")
            print(f"  Positive samples: {len(self.positive_set):,}")
            print(f"  Negative samples: {len(self.negative_set):,}")
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Create and train model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            # Can add other models here (SVM, RandomForest, etc.)
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
        
        # Train the model
        self.model.fit(X, y)
        
        # Calculate training accuracy
        if self.verbose:
            train_acc = self.model.score(X, y)
            print(f"  Training accuracy: {train_acc:.4f}")
    
    def _find_false_negatives(self) -> List[Any]:
        """
        Find items from positive set that the model incorrectly classifies.
        
        Returns:
            List of false negative items
        """
        false_negatives = []
        
        for item in self.positive_set:
            features = self._extract_features(item)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction probability
            prob = self.model.predict_proba(features_scaled)[0, 1]
            
            # Check if it's a false negative
            if prob < self.threshold:
                false_negatives.append(item)
        
        if self.verbose:
            fn_rate = len(false_negatives) / len(self.positive_set)
            print(f"  False negative rate: {fn_rate:.4f}")
            print(f"  False negatives: {len(false_negatives):,}")
        
        return false_negatives
    
    def _create_backup_filter(self):
        """Create backup filter for false negatives."""
        false_negatives = self._find_false_negatives()
        
        if len(false_negatives) == 0:
            # No false negatives, create minimal backup filter
            self.backup_filter = StandardBloomFilter(1, self.target_fpr)
            self.has_false_negatives = False
        else:
            # Create backup filter sized for false negatives
            self.backup_filter = StandardBloomFilter(
                len(false_negatives), 
                self.target_fpr,
                verbose=False
            )
            
            # Add all false negatives to backup filter
            for item in false_negatives:
                self.backup_filter.add(item)
            
            self.has_false_negatives = True
        
        if self.verbose:
            print(f"  Backup filter size: {self.backup_filter.get_memory_usage():,} bytes")
    
    def query(self, item: Any) -> bool:
        """
        Check if an item might be in the set.
        
        Args:
            item: Item to check
            
        Returns:
            True if item might be in set (possible false positive)
            False if item is definitely not in set (no false negatives)
        """
        self.query_count += 1
        
        # Extract and scale features
        features = self._extract_features(item)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get ML model prediction
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Check against threshold
        if probability >= self.threshold:
            self.positive_predictions += 1
            return True
        
        # If model says no, check backup filter
        return self.backup_filter.query(item)
    
    def __contains__(self, item: Any) -> bool:
        """Allow use of 'in' operator."""
        return self.query(item)
    
    def batch_query(self, items: List[Any]) -> List[bool]:
        """
        Query multiple items efficiently.
        
        Args:
            items: List of items to query
            
        Returns:
            List of boolean results
        """
        # Extract features for all items
        features_list = [self._extract_features(item) for item in items]
        features_matrix = np.vstack(features_list)
        features_scaled = self.scaler.transform(features_matrix)
        
        # Get batch predictions
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Check each item
        results = []
        for i, (item, prob) in enumerate(zip(items, probabilities)):
            if prob >= self.threshold:
                results.append(True)
            else:
                results.append(self.backup_filter.query(item))
        
        self.query_count += len(items)
        self.positive_predictions += sum(results)
        
        return results
    
    def get_memory_usage(self) -> dict:
        """
        Get memory usage breakdown.
        
        Returns:
            Dictionary with memory usage statistics
        """
        # Model size estimation (rough)
        model_size = 0
        if hasattr(self.model, 'coef_'):
            model_size += self.model.coef_.nbytes
        if hasattr(self.model, 'intercept_'):
            model_size += self.model.intercept_.nbytes
        
        # Scaler size
        scaler_size = 0
        if hasattr(self.scaler, 'mean_'):
            scaler_size += self.scaler.mean_.nbytes
        if hasattr(self.scaler, 'scale_'):
            scaler_size += self.scaler.scale_.nbytes
        
        return {
            'model_bytes': model_size,
            'scaler_bytes': scaler_size,
            'backup_filter_bytes': self.backup_filter.get_memory_usage(),
            'total_bytes': model_size + scaler_size + self.backup_filter.get_memory_usage()
        }
    
    def estimate_fpr(self, test_negatives: List[Any]) -> float:
        """
        Estimate false positive rate on test set.
        
        Args:
            test_negatives: Negative items to test
            
        Returns:
            Estimated false positive rate
        """
        false_positives = sum(
            1 for item in test_negatives 
            if self.query(item)
        )
        return false_positives / len(test_negatives) if test_negatives else 0.0
    
    def get_stats(self) -> dict:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary containing filter statistics
        """
        memory = self.get_memory_usage()
        
        return {
            'positive_set_size': len(self.positive_set),
            'negative_set_size': len(self.negative_set),
            'threshold': self.threshold,
            'target_fpr': self.target_fpr,
            'has_false_negatives': self.has_false_negatives,
            'backup_filter_items': self.backup_filter.count,
            'model_type': self.model_type,
            'total_queries': self.query_count,
            'positive_predictions': self.positive_predictions,
            'memory_usage': memory
        }
    
    def _print_statistics(self):
        """Print detailed statistics about the filter."""
        print("\nLearned Bloom Filter Statistics:")
        stats = self.get_stats()
        memory = stats['memory_usage']
        
        print(f"  Model type: {stats['model_type']}")
        print(f"  Threshold: {stats['threshold']:.2f}")
        print(f"  Backup filter items: {stats['backup_filter_items']:,}")
        print(f"\nMemory Usage:")
        print(f"  Model: {memory['model_bytes']:,} bytes")
        print(f"  Scaler: {memory['scaler_bytes']:,} bytes")
        print(f"  Backup filter: {memory['backup_filter_bytes']:,} bytes")
        print(f"  Total: {memory['total_bytes']:,} bytes")
        
        # Compare with standard BF
        std_bf = StandardBloomFilter(
            len(self.positive_set), 
            self.target_fpr,
            verbose=False
        )
        std_memory = std_bf.get_memory_usage()
        reduction = (1 - memory['total_bytes'] / std_memory) * 100
        
        print(f"\nComparison with Standard Bloom Filter:")
        print(f"  Standard BF: {std_memory:,} bytes")
        print(f"  Learned BF: {memory['total_bytes']:,} bytes")
        print(f"  Memory reduction: {reduction:.1f}%")
    
    def save(self, filepath: str):
        """
        Save the filter to disk.
        
        Args:
            filepath: Path to save the filter
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BasicLearnedBloomFilter':
        """
        Load a filter from disk.
        
        Args:
            filepath: Path to load the filter from
            
        Returns:
            Loaded LearnedBloomFilter instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        """String representation of the filter."""
        return (f"BasicLearnedBloomFilter(positive={len(self.positive_set)}, "
                f"negative={len(self.negative_set)}, "
                f"threshold={self.threshold:.2f}, "
                f"backup_items={self.backup_filter.count})")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Positive set (items that should be in the filter)
    positive_set = [f"valid_item_{i}" for i in range(1000)]
    
    # Negative set (items that should not be in the filter)
    negative_set = [f"invalid_item_{i}" for i in range(5000)]
    
    # Create Learned Bloom Filter
    print("Creating Learned Bloom Filter...")
    lbf = BasicLearnedBloomFilter(
        positive_set=positive_set,
        negative_set=negative_set,
        target_fpr=0.01,
        threshold=0.5,
        verbose=True
    )
    
    # Test queries
    print("\nTesting queries...")
    
    # Test positive items (should all be found)
    test_positive = positive_set[:100]
    positive_found = sum(1 for item in test_positive if lbf.query(item))
    print(f"Positive items found: {positive_found}/{len(test_positive)}")
    
    # Test negative items (should mostly not be found)
    test_negative = [f"test_negative_{i}" for i in range(1000)]
    false_positives = sum(1 for item in test_negative if lbf.query(item))
    empirical_fpr = false_positives / len(test_negative)
    print(f"Empirical FPR: {empirical_fpr:.4f}")
    
    # Batch query test
    print("\nBatch query test...")
    batch_items = test_positive[:10] + test_negative[:10]
    batch_results = lbf.batch_query(batch_items)
    print(f"Batch results: {sum(batch_results)}/{len(batch_results)} positive")