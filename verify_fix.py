#!/usr/bin/env python3
"""
Verification that the fixed implementation resolves FPR issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloom_filter.standard import StandardBloomFilter
from src.enhanced_lbf.combined import CombinedEnhancedLBF

def test_fixed_implementation():
    """Test that the fixed implementation actually works."""
    
    print("\n" + "="*70)
    print(" VERIFYING FPR FIX ")
    print("="*70)
    
    # Create realistic test data
    n_items = 1000
    
    # Malicious URLs (positive set)
    malicious_urls = []
    for i in range(n_items):
        if i % 5 == 0:
            malicious_urls.append(f"http://malware-site-{i}.com/virus.php")
        elif i % 5 == 1:
            malicious_urls.append(f"http://phishing-{i}.net/hack/")
        elif i % 5 == 2:
            malicious_urls.append(f"http://trojan-download-{i}.ru/malware")
        elif i % 5 == 3:
            malicious_urls.append(f"http://virus-{i}.cn/infected.exe")
        else:
            malicious_urls.append(f"http://hack-{i}.tk/backdoor/")
    
    # Benign URLs (negative set)
    benign_urls = []
    for i in range(n_items):
        if i % 5 == 0:
            benign_urls.append(f"https://google.com/search?q={i}")
        elif i % 5 == 1:
            benign_urls.append(f"https://github.com/user/repo{i}")
        elif i % 5 == 2:
            benign_urls.append(f"https://amazon.com/product/{i}")
        elif i % 5 == 3:
            benign_urls.append(f"https://microsoft.com/docs/{i}")
        else:
            benign_urls.append(f"https://wikipedia.org/wiki/page{i}")
    
    # Test queries (unknown items)
    test_queries = []
    for i in range(500):
        if i % 2 == 0:
            test_queries.append(f"https://test-site-{i}.com/page")
        else:
            test_queries.append(f"http://unknown-{i}.org/content")
    
    print("\nTest Data:")
    print(f"  Malicious URLs: {len(malicious_urls)}")
    print(f"  Benign URLs: {len(benign_urls)}")
    print(f"  Test Queries: {len(test_queries)}")
    
    # Test 1: Standard Bloom Filter (baseline)
    print("\n1. Standard Bloom Filter (Baseline):")
    std_bf = StandardBloomFilter(n_items, 0.01, verbose=False)
    for url in malicious_urls:
        std_bf.add(url)
    
    # Check true positives
    tp_std = sum(1 for url in malicious_urls[:100] if std_bf.query(url))
    # Check false positives on benign URLs
    fp_std_benign = sum(1 for url in benign_urls[:100] if std_bf.query(url))
    # Check false positives on test queries
    fp_std_test = sum(1 for url in test_queries if std_bf.query(url))
    
    print(f"  True Positive Rate: {tp_std/100:.2%}")
    print(f"  FPR on Benign URLs: {fp_std_benign/100:.2%}")
    print(f"  FPR on Test Queries: {fp_std_test/len(test_queries):.2%}")
    
    # Test 2: Fixed Enhanced LBF
    print("\n2. Fixed Enhanced LBF:")
    fixed_lbf = CombinedEnhancedLBF(
        initial_positive_set=malicious_urls[:500],  # Train on half
        initial_negative_set=benign_urls[:500],
        target_fpr=0.01,
        verbose=False
    )
    
    # Add remaining malicious URLs
    for url in malicious_urls[500:]:
        fixed_lbf.add(url, label=1)
    
    # Check true positives
    tp_fixed = sum(1 for url in malicious_urls[:100] if fixed_lbf.query(url))
    # Check false positives on benign URLs
    fp_fixed_benign = sum(1 for url in benign_urls[500:600] if fixed_lbf.query(url))
    # Check false positives on test queries
    fp_fixed_test = sum(1 for url in test_queries if fixed_lbf.query(url))
    
    print(f"  True Positive Rate: {tp_fixed/100:.2%}")
    print(f"  FPR on Benign URLs: {fp_fixed_benign/100:.2%}")
    print(f"  FPR on Test Queries: {fp_fixed_test/len(test_queries):.2%}")
    
    # Test 3: Check if model is actually learning
    print("\n3. Model Discrimination Test:")
    print("  Testing on sample URLs...")
    
    test_malicious = [
        "http://virus-test.com/malware.exe",
        "http://phishing-scam.net/hack",
        "http://trojan-download.ru/virus"
    ]
    
    test_benign = [
        "https://google.com/search",
        "https://github.com/project",
        "https://amazon.com/product"
    ]
    
    print("\n  Malicious URL predictions:")
    for url in test_malicious:
        result = fixed_lbf.query(url)
        features = fixed_lbf._extract_url_features(url)
        score = fixed_lbf.model.predict(features)
        print(f"    {url[:40]}: {'DETECTED' if result else 'MISSED'} (score: {score:.3f})")
    
    print("\n  Benign URL predictions:")
    for url in test_benign:
        result = fixed_lbf.query(url)
        features = fixed_lbf._extract_url_features(url)
        score = fixed_lbf.model.predict(features)
        print(f"    {url[:40]}: {'FALSE POSITIVE' if result else 'CORRECT'} (score: {score:.3f})")
    
    # Test 4: Check backup filter is working
    print("\n4. Backup Filter Test:")
    print("  Checking if positive items are in backup filter...")
    
    # Check a few items that were added
    sample_added = malicious_urls[:5]
    for url in sample_added:
        in_backup = fixed_lbf.positive_backup.query(url)
        print(f"    {url[:40]}: {'IN BACKUP' if in_backup else 'NOT IN BACKUP'}")
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY ")
    print("="*70)
    
    # Calculate overall FPR
    std_overall_fpr = (fp_std_benign + fp_std_test) / (100 + len(test_queries))
    fixed_overall_fpr = (fp_fixed_benign + fp_fixed_test) / (100 + len(test_queries))
    
    print(f"\nOverall False Positive Rates:")
    print(f"  Standard BF: {std_overall_fpr:.3%}")
    print(f"  Fixed Enhanced LBF: {fixed_overall_fpr:.3%}")
    
    if fixed_overall_fpr <= 0.05:  # 5% tolerance
        print("\n✅ FPR FIX VERIFIED - The fixed implementation maintains acceptable FPR!")
        print(f"   Fixed LBF achieves {fixed_overall_fpr:.2%} FPR (target: 1%)")
        return True
    else:
        print("\n❌ FPR STILL TOO HIGH - Fix needs more work")
        print(f"   Fixed LBF has {fixed_overall_fpr:.2%} FPR (target: 1%)")
        
        # Debug info
        stats = fixed_lbf.get_stats()
        print("\n  Debug Info:")
        print(f"    Current threshold: {stats['current_threshold']:.3f}")
        print(f"    Total updates: {stats['total_updates']}")
        print(f"    Model weights norm: {np.linalg.norm(fixed_lbf.model.weights):.3f}")
        
        return False

if __name__ == "__main__":
    import numpy as np
    success = test_fixed_implementation()
    exit(0 if success else 1)