"""
Standard Bloom Filter Implementation

This serves as the baseline for comparison with our enhanced Learned Bloom Filter.
Implements optimal parameter calculation and standard add/query operations.
"""

import math
import mmh3
from bitarray import bitarray
from typing import Any, List, Callable
import hashlib


class StandardBloomFilter:
    """
    A standard Bloom Filter implementation with optimal parameter calculation.
    
    Attributes:
        m: Number of bits in the filter
        k: Number of hash functions
        n: Expected number of elements
        false_positive_rate: Target false positive rate
    """
    
    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01, verbose: bool = False):
        """
        Initialize a Bloom Filter with optimal parameters.
        
        Args:
            expected_elements: Expected number of elements to add
            false_positive_rate: Desired false positive rate (default 0.01 = 1%)
        """
        if expected_elements <= 0:
            raise ValueError("Expected elements must be positive")
        if not 0 < false_positive_rate < 1:
            raise ValueError("False positive rate must be between 0 and 1")
            
        self.n = expected_elements
        self.p = false_positive_rate
        
        # Calculate optimal bit array size (m) and number of hash functions (k)
        self.m = self._calculate_m(expected_elements, false_positive_rate)
        self.k = self._calculate_k(self.m, expected_elements)
        
        # Initialize bit array
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)
        
        # Count of actual elements added
        self.count = 0
        
        # Create hash functions
        self.hash_functions = self._create_hash_functions(self.k)
        
        if verbose:
            print(f"Initialized StandardBloomFilter:")
            print(f"  Expected elements (n): {self.n:,}")
            print(f"  Target FPR (p): {self.p:.4f}")
            print(f"  Bit array size (m): {self.m:,} bits ({self.m / 8 / 1024:.2f} KB)")
            print(f"  Hash functions (k): {self.k}")
            print(f"  Bits per element: {self.m / self.n:.2f}")
    
    @staticmethod
    def _calculate_m(n: int, p: float) -> int:
        """
        Calculate optimal bit array size.
        Formula: m = -n * ln(p) / (ln(2)^2)
        """
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))
    
    @staticmethod
    def _calculate_k(m: int, n: int) -> int:
        """
        Calculate optimal number of hash functions.
        Formula: k = (m/n) * ln(2)
        """
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))
    
    def _create_hash_functions(self, k: int) -> List[Callable]:
        """
        Create k independent hash functions using MurmurHash3 with different seeds.
        """
        def make_hash_func(seed):
            return lambda item: mmh3.hash(str(item), seed, signed=False) % self.m
        
        return [make_hash_func(i) for i in range(k)]
    
    def add(self, item: Any) -> None:
        """
        Add an item to the Bloom Filter.
        
        Args:
            item: Item to add (will be converted to string for hashing)
        """
        for hash_func in self.hash_functions:
            index = hash_func(item)
            self.bit_array[index] = 1
        self.count += 1
    
    def query(self, item: Any) -> bool:
        """
        Check if an item might be in the set.
        
        Args:
            item: Item to check
            
        Returns:
            True if item might be in set (possible false positive)
            False if item is definitely not in set (no false negatives)
        """
        for hash_func in self.hash_functions:
            index = hash_func(item)
            if not self.bit_array[index]:
                return False
        return True
    
    def __contains__(self, item: Any) -> bool:
        """Allow use of 'in' operator."""
        return self.query(item)
    
    def add_all(self, items: List[Any]) -> None:
        """
        Add multiple items to the filter.
        
        Args:
            items: List of items to add
        """
        for item in items:
            self.add(item)
    
    def get_load_factor(self) -> float:
        """
        Calculate the current load factor (proportion of bits set to 1).
        
        Returns:
            Float between 0 and 1 representing the load factor
        """
        return self.bit_array.count(1) / self.m
    
    def estimate_fpr(self) -> float:
        """
        Estimate the current false positive rate based on load factor.
        Formula: (1 - e^(-kn/m))^k
        
        Returns:
            Estimated false positive rate
        """
        if self.count == 0:
            return 0.0
        
        # Use actual count instead of expected n
        return (1 - math.exp(-self.k * self.count / self.m)) ** self.k
    
    def clear(self) -> None:
        """Reset the filter to empty state."""
        self.bit_array.setall(0)
        self.count = 0
    
    def union(self, other: 'StandardBloomFilter') -> 'StandardBloomFilter':
        """
        Create a new filter that is the union of this and another filter.
        Both filters must have the same m and k parameters.
        
        Args:
            other: Another StandardBloomFilter
            
        Returns:
            New StandardBloomFilter containing union
        """
        if self.m != other.m or self.k != other.k:
            raise ValueError("Cannot union filters with different parameters")
        
        result = StandardBloomFilter.__new__(StandardBloomFilter)
        result.m = self.m
        result.k = self.k
        result.n = self.n + other.n
        result.p = self.p
        result.hash_functions = self.hash_functions
        result.bit_array = self.bit_array | other.bit_array
        result.count = self.count + other.count  # Approximate
        
        return result
    
    def intersection(self, other: 'StandardBloomFilter') -> 'StandardBloomFilter':
        """
        Create a new filter that is the intersection of this and another filter.
        Note: This is approximate and may have higher FPR.
        
        Args:
            other: Another StandardBloomFilter
            
        Returns:
            New StandardBloomFilter containing intersection
        """
        if self.m != other.m or self.k != other.k:
            raise ValueError("Cannot intersect filters with different parameters")
        
        result = StandardBloomFilter.__new__(StandardBloomFilter)
        result.m = self.m
        result.k = self.k
        result.n = min(self.n, other.n)
        result.p = self.p
        result.hash_functions = self.hash_functions
        result.bit_array = self.bit_array & other.bit_array
        result.count = min(self.count, other.count)  # Approximate
        
        return result
    
    def get_memory_usage(self) -> int:
        """
        Get memory usage in bytes.
        
        Returns:
            Memory usage in bytes
        """
        return self.m // 8 + (1 if self.m % 8 else 0)
    
    def get_stats(self) -> dict:
        """
        Get comprehensive statistics about the filter.
        
        Returns:
            Dictionary containing filter statistics
        """
        return {
            'expected_elements': self.n,
            'actual_elements': self.count,
            'target_fpr': self.p,
            'estimated_fpr': self.estimate_fpr(),
            'bits': self.m,
            'hash_functions': self.k,
            'load_factor': self.get_load_factor(),
            'memory_bytes': self.get_memory_usage(),
            'bits_per_element': self.m / self.n if self.n > 0 else 0
        }
    
    def __repr__(self) -> str:
        """String representation of the filter."""
        return (f"StandardBloomFilter(n={self.n}, p={self.p:.4f}, "
                f"m={self.m}, k={self.k}, count={self.count})")


# Example usage and testing
if __name__ == "__main__":
    # Create a Bloom Filter for 10,000 elements with 1% FPR
    bf = StandardBloomFilter(expected_elements=10000, false_positive_rate=0.01, verbose=True)
    
    # Add some items
    items = [f"item_{i}" for i in range(5000)]
    bf.add_all(items)
    
    # Test membership
    print(f"\nTesting membership:")
    print(f"'item_100' in filter: {bf.query('item_100')}")  # Should be True
    print(f"'item_9999' in filter: {bf.query('item_9999')}")  # Should be False
    print(f"'not_added' in filter: {bf.query('not_added')}")  # Should be False (probably)
    
    # Check statistics
    print(f"\nFilter statistics:")
    stats = bf.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")
    
    # Test false positive rate empirically
    false_positives = 0
    test_items = 10000
    for i in range(5000, 5000 + test_items):
        if bf.query(f"item_{i}"):
            false_positives += 1
    
    empirical_fpr = false_positives / test_items
    print(f"\nEmpirical FPR: {empirical_fpr:.4f} (tested with {test_items:,} items)")