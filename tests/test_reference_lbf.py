"""
Tests for reference Learned Bloom Filter implementations.

Verifies correctness before using in benchmarks.
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reference_lbf.ada_bf import AdaptiveLearnedBloomFilter
from src.reference_lbf.plbf import PartitionedLearnedBloomFilter
from src.reference_lbf.stable_lbf import StableLearnedBloomFilter


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    positive_samples = [
        f"http://malicious-site-{i}.com/evil" for i in range(100)
    ]
    negative_samples = [
        f"http://legitimate-site-{i}.com/normal" for i in range(100)
    ]
    return positive_samples, negative_samples


class TestAdaptiveLBF:
    """Test Adaptive Learned Bloom Filter (Ada-BF)."""
    
    def test_initialization(self, sample_data):
        """Test that Ada-BF initializes correctly."""
        positive, negative = sample_data
        
        ada_bf = AdaptiveLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01,
            n_regions=5,
            verbose=True
        )
        
        assert ada_bf.target_fpr == 0.01
        assert ada_bf.n_regions == 5
        assert len(ada_bf.positive_set) == 100
        assert len(ada_bf.backup_filters) == 5
    
    def test_query_positives(self, sample_data):
        """Test that all positive samples return True."""
        positive, negative = sample_data
        
        ada_bf = AdaptiveLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01
        )
        
        # All positive samples should return True (no false negatives)
        for item in positive:
            assert ada_bf.query(item), f"False negative for: {item}"
    
    def test_query_negatives(self, sample_data):
        """Test FPR on negative samples."""
        positive, negative = sample_data
        
        ada_bf = AdaptiveLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01
        )
        
        # Count false positives
        false_positives = sum(1 for item in negative if ada_bf.query(item))
        fpr = false_positives / len(negative)
        
        # FPR should be reasonably close to target (with tolerance)
        assert fpr <= 0.1, f"FPR too high: {fpr:.3f}"
    
    def test_add_functionality(self, sample_data):
        """Test adding new items."""
        positive, negative = sample_data
        
        ada_bf = AdaptiveLearnedBloomFilter(
            positive_set=positive[:50],
            negative_set=negative,
            target_fpr=0.01
        )
        
        # Add remaining positives
        for item in positive[50:]:
            ada_bf.add(item)
        
        # All should be queryable
        for item in positive:
            assert ada_bf.query(item)
    
    def test_statistics(self, sample_data):
        """Test statistics collection."""
        positive, negative = sample_data
        
        ada_bf = AdaptiveLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01
        )
        
        # Perform some queries
        for item in positive[:10]:
            ada_bf.query(item)
        
        stats = ada_bf.get_stats()
        assert stats['total_queries'] == 10
        assert stats['positive_count'] == 100
        assert stats['memory_kb'] > 0


class TestPartitionedLBF:
    """Test Partitioned Learned Bloom Filter (PLBF)."""
    
    def test_initialization(self, sample_data):
        """Test that PLBF initializes correctly."""
        positive, negative = sample_data
        
        plbf = PartitionedLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01,
            n_partitions=4,
            verbose=True
        )
        
        assert plbf.target_fpr == 0.01
        assert plbf.n_partitions == 4
        assert len(plbf.positive_set) == 100
        assert len(plbf.backup_filters) == 4
    
    def test_query_positives(self, sample_data):
        """Test that all positive samples return True."""
        positive, negative = sample_data
        
        plbf = PartitionedLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01
        )
        
        # All positive samples should return True
        for item in positive:
            assert plbf.query(item), f"False negative for: {item}"
    
    def test_partitioning(self, sample_data):
        """Test that items are properly partitioned."""
        positive, negative = sample_data
        
        plbf = PartitionedLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01,
            n_partitions=4
        )
        
        # Check that partitions exist and have items
        assert len(plbf.partitions) == 4
        total_items = sum(len(p) for p in plbf.partitions)
        assert total_items == 100
    
    def test_memory_usage(self, sample_data):
        """Test memory usage reporting."""
        positive, negative = sample_data
        
        plbf = PartitionedLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01
        )
        
        memory = plbf.get_memory_usage()
        assert memory > 0
        assert memory < 1_000_000  # Sanity check


class TestStableLBF:
    """Test Stable Learned Bloom Filter (s-SLBF)."""
    
    def test_initialization(self, sample_data):
        """Test that Stable LBF initializes correctly."""
        positive, negative = sample_data
        
        slbf = StableLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01,
            retrain_threshold=50,
            verbose=True
        )
        
        assert slbf.target_fpr == 0.01
        assert slbf.retrain_threshold == 50
        assert len(slbf.positive_set) == 100
    
    def test_query_positives(self, sample_data):
        """Test that all positive samples return True."""
        positive, negative = sample_data
        
        slbf = StableLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01
        )
        
        # All positive samples should return True
        for item in positive:
            assert slbf.query(item), f"False negative for: {item}"
    
    def test_streaming_insertions(self, sample_data):
        """Test continuous insertions (key feature of Stable LBF)."""
        positive, negative = sample_data
        
        slbf = StableLearnedBloomFilter(
            positive_set=positive[:50],
            negative_set=negative,
            target_fpr=0.01,
            retrain_threshold=20
        )
        
        # Add items continuously
        for item in positive[50:70]:
            slbf.add(item)
        
        # All should be queryable
        for item in positive[:70]:
            assert slbf.query(item)
        
        # Should have retrained at least once
        assert slbf.retrain_count >= 1
    
    def test_retrain_trigger(self, sample_data):
        """Test that retraining is triggered correctly."""
        positive, negative = sample_data
        
        slbf = StableLearnedBloomFilter(
            positive_set=positive[:50],
            negative_set=negative,
            target_fpr=0.01,
            retrain_threshold=10
        )
        
        initial_retrain_count = slbf.retrain_count
        
        # Add items to trigger retrain
        for i in range(15):
            slbf.add(f"new-item-{i}")
        
        # Retrain should have been triggered
        assert slbf.retrain_count > initial_retrain_count
    
    def test_statistics(self, sample_data):
        """Test statistics collection."""
        positive, negative = sample_data
        
        slbf = StableLearnedBloomFilter(
            positive_set=positive,
            negative_set=negative,
            target_fpr=0.01
        )
        
        # Perform operations
        for item in positive[:10]:
            slbf.query(item)
        
        slbf.add("new-item")
        
        stats = slbf.get_stats()
        assert stats['total_queries'] == 10
        assert stats['total_insertions'] == 1
        assert stats['positive_count'] == 101


def test_all_implementations_basic():
    """Smoke test for all implementations."""
    positive = [f"pos-{i}" for i in range(50)]
    negative = [f"neg-{i}" for i in range(50)]
    
    # Ada-BF
    ada = AdaptiveLearnedBloomFilter(positive, negative)
    assert all(ada.query(item) for item in positive)
    
    # PLBF
    plbf = PartitionedLearnedBloomFilter(positive, negative)
    assert all(plbf.query(item) for item in positive)
    
    # Stable LBF
    slbf = StableLearnedBloomFilter(positive, negative)
    assert all(slbf.query(item) for item in positive)
    
    print("\nAll reference implementations passed smoke test!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
