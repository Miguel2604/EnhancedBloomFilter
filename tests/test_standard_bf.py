"""
Unit tests for Standard Bloom Filter implementation
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bloom_filter.standard import StandardBloomFilter


class TestStandardBloomFilter:
    """Test suite for Standard Bloom Filter."""
    
    def test_initialization(self):
        """Test filter initialization with various parameters."""
        # Normal initialization
        bf = StandardBloomFilter(1000, 0.01)
        assert bf.n == 1000
        assert bf.p == 0.01
        assert bf.count == 0
        assert bf.m > 0
        assert bf.k > 0
        
        # Different FPR
        bf2 = StandardBloomFilter(1000, 0.001)
        assert bf2.m > bf.m  # Smaller FPR needs more bits
        assert bf2.k > bf.k  # And more hash functions
    
    def test_invalid_initialization(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            StandardBloomFilter(0, 0.01)  # Zero elements
        
        with pytest.raises(ValueError):
            StandardBloomFilter(-100, 0.01)  # Negative elements
        
        with pytest.raises(ValueError):
            StandardBloomFilter(1000, 0)  # Zero FPR
        
        with pytest.raises(ValueError):
            StandardBloomFilter(1000, 1)  # 100% FPR
        
        with pytest.raises(ValueError):
            StandardBloomFilter(1000, 1.5)  # FPR > 1
    
    def test_add_and_query(self):
        """Test adding elements and querying membership."""
        bf = StandardBloomFilter(100, 0.01)
        
        # Add elements
        bf.add("hello")
        bf.add("world")
        bf.add(123)
        bf.add(45.67)
        
        # Test membership (should all be True)
        assert bf.query("hello") is True
        assert bf.query("world") is True
        assert bf.query(123) is True
        assert bf.query(45.67) is True
        
        # Test non-membership (should be False, but might have false positives)
        # These are likely to be False but not guaranteed
        assert bf.query("not_added") in [True, False]
        assert bf.query(999) in [True, False]
    
    def test_no_false_negatives(self):
        """Test that there are no false negatives."""
        bf = StandardBloomFilter(1000, 0.01)
        items = [f"item_{i}" for i in range(100)]
        
        # Add all items
        for item in items:
            bf.add(item)
        
        # All added items must be found (no false negatives)
        for item in items:
            assert bf.query(item) is True
    
    def test_false_positive_rate(self):
        """Test that FPR is close to target."""
        target_fpr = 0.01
        bf = StandardBloomFilter(10000, target_fpr)
        
        # Add first half of items
        n_add = 5000
        for i in range(n_add):
            bf.add(f"item_{i}")
        
        # Test second half (not added)
        false_positives = 0
        n_test = 5000
        for i in range(n_add, n_add + n_test):
            if bf.query(f"item_{i}"):
                false_positives += 1
        
        empirical_fpr = false_positives / n_test
        
        # FPR should be reasonably close to target (within 3x)
        # Note: This is probabilistic, so we allow some variance
        assert empirical_fpr < target_fpr * 3
    
    def test_contains_operator(self):
        """Test the __contains__ method (in operator)."""
        bf = StandardBloomFilter(100, 0.01)
        bf.add("test")
        
        assert "test" in bf
        # "not_in" might have false positive, so we can't assert it's False
    
    def test_add_all(self):
        """Test bulk addition of items."""
        bf = StandardBloomFilter(1000, 0.01)
        items = ["a", "b", "c", "d", "e"]
        
        bf.add_all(items)
        
        for item in items:
            assert bf.query(item) is True
        
        assert bf.count == len(items)
    
    def test_clear(self):
        """Test clearing the filter."""
        bf = StandardBloomFilter(100, 0.01)
        bf.add("item1")
        bf.add("item2")
        
        assert bf.query("item1") is True
        assert bf.count == 2
        
        bf.clear()
        
        assert bf.count == 0
        assert bf.get_load_factor() == 0.0
        # After clear, items should not be found (unless false positive)
        # But we can't guarantee they're not found due to hash collisions
    
    def test_load_factor(self):
        """Test load factor calculation."""
        bf = StandardBloomFilter(1000, 0.01)
        
        initial_load = bf.get_load_factor()
        assert initial_load == 0.0
        
        # Add items and check load factor increases
        for i in range(100):
            bf.add(f"item_{i}")
        
        final_load = bf.get_load_factor()
        assert final_load > initial_load
        assert 0 < final_load < 1
    
    def test_estimate_fpr(self):
        """Test FPR estimation."""
        bf = StandardBloomFilter(1000, 0.01)
        
        # Empty filter should have 0 FPR
        assert bf.estimate_fpr() == 0.0
        
        # Add items and check FPR increases
        for i in range(500):
            bf.add(f"item_{i}")
        
        estimated_fpr = bf.estimate_fpr()
        assert 0 < estimated_fpr < 1
    
    def test_union(self):
        """Test union of two filters."""
        bf1 = StandardBloomFilter(1000, 0.01)
        bf2 = StandardBloomFilter(1000, 0.01)
        
        # Add different items to each
        for i in range(50):
            bf1.add(f"bf1_item_{i}")
            bf2.add(f"bf2_item_{i}")
        
        # Create union
        bf_union = bf1.union(bf2)
        
        # Both sets of items should be in union
        for i in range(50):
            assert bf_union.query(f"bf1_item_{i}") is True
            assert bf_union.query(f"bf2_item_{i}") is True
    
    def test_union_incompatible(self):
        """Test that union fails with incompatible filters."""
        bf1 = StandardBloomFilter(1000, 0.01)
        bf2 = StandardBloomFilter(2000, 0.01)  # Different size
        
        with pytest.raises(ValueError):
            bf1.union(bf2)
    
    def test_intersection(self):
        """Test intersection of two filters."""
        bf1 = StandardBloomFilter(1000, 0.01)
        bf2 = StandardBloomFilter(1000, 0.01)
        
        # Add overlapping items
        for i in range(50):
            bf1.add(f"common_item_{i}")
            bf2.add(f"common_item_{i}")
        
        # Add unique items
        for i in range(50):
            bf1.add(f"bf1_only_{i}")
            bf2.add(f"bf2_only_{i}")
        
        # Create intersection
        bf_intersect = bf1.intersection(bf2)
        
        # Common items should likely be in intersection
        # (not guaranteed due to approximate nature)
        common_found = sum(
            1 for i in range(50) 
            if bf_intersect.query(f"common_item_{i}")
        )
        assert common_found > 40  # Most should be found
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        bf = StandardBloomFilter(1000, 0.01)
        memory = bf.get_memory_usage()
        
        assert memory > 0
        assert memory == (bf.m // 8) + (1 if bf.m % 8 else 0)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        bf = StandardBloomFilter(1000, 0.01)
        
        # Add some items
        for i in range(100):
            bf.add(f"item_{i}")
        
        stats = bf.get_stats()
        
        assert stats['expected_elements'] == 1000
        assert stats['actual_elements'] == 100
        assert stats['target_fpr'] == 0.01
        assert 'estimated_fpr' in stats
        assert stats['bits'] == bf.m
        assert stats['hash_functions'] == bf.k
        assert 'load_factor' in stats
        assert 'memory_bytes' in stats
        assert 'bits_per_element' in stats
    
    def test_repr(self):
        """Test string representation."""
        bf = StandardBloomFilter(1000, 0.01)
        repr_str = repr(bf)
        
        assert "StandardBloomFilter" in repr_str
        assert "n=1000" in repr_str
        assert "p=0.01" in repr_str


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])