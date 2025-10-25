#!/usr/bin/env python3
"""
Verify Testing Methodology for Enhanced LBF

This script checks for potential testing issues:
1. Data leakage (overlap between training and test sets)
2. Proper separation of train/test data
3. Legitimate FPR measurements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.enhanced_lbf.combined import CombinedEnhancedLBF
from src.bloom_filter.standard import StandardBloomFilter
import random
import time

def load_url_dataset():
    """Load real URL dataset."""
    dataset_path = "data/datasets/url_blacklist"
    
    malicious = []
    benign = []
    
    malicious_file = f"{dataset_path}/malicious_urls.txt"
    benign_file = f"{dataset_path}/benign_urls.txt"
    
    if os.path.exists(malicious_file):
        with open(malicious_file, 'r') as f:
            malicious = [line.strip() for line in f.readlines() if line.strip()]
    
    if os.path.exists(benign_file):
        with open(benign_file, 'r') as f:
            benign = [line.strip() for line in f.readlines() if line.strip()]
    
    return malicious, benign


def test_current_methodology():
    """Test with current benchmark methodology (has data leakage)."""
    print("="*80)
    print("TEST 1: Current Methodology (AS USED IN BENCHMARK)")
    print("="*80)
    
    malicious, benign = load_url_dataset()
    
    positive_set = malicious[:10000]
    negative_set = benign[:10000]
    
    # CURRENT METHOD (from benchmark)
    train_size = min(1000, len(positive_set) // 5)
    train_negative_size = min(1000, len(negative_set) // 5)
    
    print(f"\nDataset sizes:")
    print(f"  Total positive: {len(positive_set)}")
    print(f"  Total negative: {len(negative_set)}")
    print(f"  Training positive: {train_size}")
    print(f"  Training negative: {train_negative_size}")
    
    # Initialize with training data
    lbf = CombinedEnhancedLBF(
        initial_positive_set=positive_set[:train_size],
        initial_negative_set=negative_set[:train_negative_size],
        target_fpr=0.01,
        verbose=False
    )
    
    # Add remaining items
    for item in positive_set[train_size:]:
        lbf.add(item, label=1)
    
    # Test queries (CURRENT METHOD - has overlap)
    query_positives = positive_set[:1000]
    query_negatives = negative_set[:1000]
    
    print(f"\nTest set sizes:")
    print(f"  Query positive: {len(query_positives)}")
    print(f"  Query negative: {len(query_negatives)}")
    
    # Check overlap
    train_set = set(positive_set[:train_size])
    test_set = set(query_positives)
    overlap = train_set.intersection(test_set)
    
    print(f"\n⚠️  DATA LEAKAGE ANALYSIS:")
    print(f"  Training set size: {len(train_set)}")
    print(f"  Test set size: {len(test_set)}")
    print(f"  Overlap: {len(overlap)} items ({len(overlap)/len(test_set)*100:.1f}%)")
    
    # Measure performance
    tp = sum(1 for item in query_positives if lbf.query(item))
    fp = sum(1 for item in query_negatives if lbf.query(item))
    
    tpr = tp / len(query_positives)
    fpr = fp / len(query_negatives)
    
    print(f"\n📊 Results:")
    print(f"  True Positive Rate: {tpr*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    
    return fpr


def test_corrected_methodology():
    """Test with proper train/test split (NO data leakage)."""
    print("\n\n" + "="*80)
    print("TEST 2: Corrected Methodology (PROPER TRAIN/TEST SPLIT)")
    print("="*80)
    
    malicious, benign = load_url_dataset()
    
    # Shuffle to ensure randomness
    random.seed(42)
    random.shuffle(malicious)
    random.shuffle(benign)
    
    # CORRECTED: Proper split
    train_split = 0.7  # 70% train, 30% test
    split_idx_pos = int(len(malicious) * train_split)
    split_idx_neg = int(len(benign) * train_split)
    
    train_positive = malicious[:split_idx_pos][:1000]  # Use first 1000 for training
    test_positive = malicious[split_idx_pos:][:1000]   # Use 1000 from test portion
    
    train_negative = benign[:split_idx_neg][:1000]
    test_negative = benign[split_idx_neg:][:1000]
    
    print(f"\nDataset sizes:")
    print(f"  Training positive: {len(train_positive)}")
    print(f"  Training negative: {len(train_negative)}")
    print(f"  Test positive: {len(test_positive)}")
    print(f"  Test negative: {len(test_negative)}")
    
    # Check overlap (should be zero)
    train_set = set(train_positive)
    test_set = set(test_positive)
    overlap = train_set.intersection(test_set)
    
    print(f"\n✅ DATA LEAKAGE ANALYSIS:")
    print(f"  Training set size: {len(train_set)}")
    print(f"  Test set size: {len(test_set)}")
    print(f"  Overlap: {len(overlap)} items ({len(overlap)/len(test_set)*100:.1f}% - SHOULD BE 0%)")
    
    # Initialize with training data only
    lbf = CombinedEnhancedLBF(
        initial_positive_set=train_positive,
        initial_negative_set=train_negative,
        target_fpr=0.01,
        verbose=False
    )
    
    # Test on UNSEEN data
    tp = sum(1 for item in test_positive if lbf.query(item))
    fp = sum(1 for item in test_negative if lbf.query(item))
    
    tpr = tp / len(test_positive)
    fpr = fp / len(test_negative)
    
    print(f"\n📊 Results:")
    print(f"  True Positive Rate: {tpr*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    
    return fpr


def test_fpr_only_on_negatives():
    """Test to verify FPR is legitimate (only negative items not in filter)."""
    print("\n\n" + "="*80)
    print("TEST 3: FPR Legitimacy Check")
    print("="*80)
    
    malicious, benign = load_url_dataset()
    
    positive_set = malicious[:1000]
    negative_set = benign[:2000]
    
    # Split negatives: some for training, some for testing
    train_negative = negative_set[:1000]
    test_negative = negative_set[1000:]
    
    print(f"\nSetup:")
    print(f"  Positive items (added to filter): {len(positive_set)}")
    print(f"  Negative items (for ML training): {len(train_negative)}")
    print(f"  Negative items (for FPR testing): {len(test_negative)}")
    
    # Check: Test negatives should NOT overlap with training negatives
    overlap = set(train_negative).intersection(set(test_negative))
    print(f"  Overlap between train and test negatives: {len(overlap)}")
    
    lbf = CombinedEnhancedLBF(
        initial_positive_set=positive_set,
        initial_negative_set=train_negative,
        target_fpr=0.01,
        verbose=False
    )
    
    # Key question: Are negative items added to the Bloom filters?
    print(f"\n🔍 Checking if negative items are in filters:")
    
    # Sample a few negative items from training
    sample_negative = train_negative[:10]
    in_primary = sum(1 for item in sample_negative if lbf.primary_filter.query(item))
    in_backup = sum(1 for item in sample_negative if lbf.positive_backup.query(item))
    
    print(f"  Negative training items in primary_filter: {in_primary}/10")
    print(f"  Negative training items in positive_backup: {in_backup}/10")
    
    # Now test FPR on unseen negatives
    fp = sum(1 for item in test_negative if lbf.query(item))
    fpr = fp / len(test_negative)
    
    print(f"\n📊 FPR on UNSEEN negative items: {fpr*100:.2f}%")
    
    # Compare with Standard BF
    sbf = StandardBloomFilter(expected_elements=len(positive_set), false_positive_rate=0.01)
    for item in positive_set:
        sbf.add(item)
    
    fp_standard = sum(1 for item in test_negative if sbf.query(item))
    fpr_standard = fp_standard / len(test_negative)
    
    print(f"📊 Standard BF FPR (for comparison): {fpr_standard*100:.2f}%")
    
    return fpr, fpr_standard


def main():
    """Run all verification tests."""
    print("\n" + "🔬 ENHANCED LBF TESTING METHODOLOGY VERIFICATION")
    print("="*80)
    
    # Test 1: Current methodology
    fpr_current = test_current_methodology()
    
    # Test 2: Corrected methodology
    fpr_corrected = test_corrected_methodology()
    
    # Test 3: FPR legitimacy
    fpr_enhanced, fpr_standard = test_fpr_only_on_negatives()
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n1. Current Benchmark Methodology:")
    print(f"   - Has data leakage: Training and test sets overlap")
    print(f"   - FPR: {fpr_current*100:.2f}%")
    print(f"   - ⚠️  TPR is artificially inflated due to overlap")
    
    print(f"\n2. Corrected Methodology (Proper Split):")
    print(f"   - No data leakage: Separate train/test sets")
    print(f"   - FPR: {fpr_corrected*100:.2f}%")
    print(f"   - ✅ True performance measure")
    
    print(f"\n3. FPR Measurement Legitimacy:")
    print(f"   - Enhanced LBF FPR: {fpr_enhanced*100:.2f}%")
    print(f"   - Standard BF FPR: {fpr_standard*100:.2f}%")
    
    if abs(fpr_current - fpr_corrected) < 0.02:
        print(f"\n✅ CONCLUSION: FPR measurements appear legitimate")
        print(f"   The low FPR is real, not due to data leakage.")
        print(f"   However, TPR measurements in benchmarks have overlap issues.")
    else:
        print(f"\n⚠️  CONCLUSION: Significant difference found")
        print(f"   FPR may be affected by testing methodology.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
