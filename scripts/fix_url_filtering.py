#!/usr/bin/env python3
"""
Diagnose and fix the URL filtering false positive issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from src.enhanced_lbf.combined import CombinedEnhancedLBF
from src.bloom_filter.standard import StandardBloomFilter


def diagnose_issue():
    """Diagnose the URL filtering problem."""
    print("\n" + "="*70)
    print(" URL FILTERING DIAGNOSTIC ")
    print("="*70)
    
    # Load dataset
    data_dir = Path("data/datasets/url_blacklist")
    
    if not data_dir.exists():
        print("⚠ Dataset not found. Please run download_datasets.py first.")
        return
    
    # Load URLs
    with open(data_dir / "malicious_urls.txt", 'r') as f:
        malicious_urls = [line.strip() for line in f.readlines()[:1000]]
    
    with open(data_dir / "benign_urls.txt", 'r') as f:
        benign_urls = [line.strip() for line in f.readlines()[:1000]]
    
    print(f"\nLoaded {len(malicious_urls)} malicious URLs")
    print(f"Loaded {len(benign_urls)} benign URLs")
    
    # Test 1: Basic model behavior
    print("\n" + "-"*50)
    print("TEST 1: Basic Model Behavior")
    print("-"*50)
    
    # Create a simple filter
    filter_bf = CombinedEnhancedLBF(
        initial_positive_set=malicious_urls[:100],
        initial_negative_set=benign_urls[:100],
        target_fpr=0.01,
        verbose=True
    )
    
    # Check predictions on training data
    print("\nChecking predictions on training data:")
    
    mal_detected = sum(1 for url in malicious_urls[:100] if filter_bf.query(url))
    print(f"  Malicious URLs detected: {mal_detected}/100 ({mal_detected}%)")
    
    ben_detected = sum(1 for url in benign_urls[:100] if filter_bf.query(url))
    print(f"  Benign URLs falsely detected: {ben_detected}/100 ({ben_detected}%)")
    
    # Test 2: Check model weights
    print("\n" + "-"*50)
    print("TEST 2: Model Weights Analysis")
    print("-"*50)
    
    print(f"Model weights: {filter_bf.model.weights}")
    print(f"Model bias: {filter_bf.model.bias}")
    print(f"Current threshold: {filter_bf.threshold}")
    
    # Test 3: Feature extraction
    print("\n" + "-"*50)
    print("TEST 3: Feature Analysis")
    print("-"*50)
    
    # Sample URLs
    sample_mal = malicious_urls[200]
    sample_ben = benign_urls[200]
    
    print(f"\nMalicious URL: {sample_mal[:50]}...")
    mal_features = filter_bf._extract_features(sample_mal)
    mal_score = filter_bf.model.predict(mal_features)
    mal_prob = 1 / (1 + np.exp(-mal_score))
    print(f"  Features: {mal_features[:5]}...")
    print(f"  Score: {mal_score:.3f}")
    print(f"  Probability: {mal_prob:.3f}")
    print(f"  Prediction: {'Malicious' if mal_prob >= filter_bf.threshold else 'Benign'}")
    
    print(f"\nBenign URL: {sample_ben[:50]}...")
    ben_features = filter_bf._extract_features(sample_ben)
    ben_score = filter_bf.model.predict(ben_features)
    ben_prob = 1 / (1 + np.exp(-ben_score))
    print(f"  Features: {ben_features[:5]}...")
    print(f"  Score: {ben_score:.3f}")
    print(f"  Probability: {ben_prob:.3f}")
    print(f"  Prediction: {'Malicious' if ben_prob >= filter_bf.threshold else 'Benign'}")
    
    # Test 4: Try with more training data
    print("\n" + "-"*50)
    print("TEST 4: Extended Training")
    print("-"*50)
    
    # Create filter with more data
    filter_bf2 = CombinedEnhancedLBF(
        initial_positive_set=malicious_urls[:500],
        initial_negative_set=benign_urls[:500],
        target_fpr=0.01,
        verbose=False
    )
    
    # Test on holdout data
    test_mal = malicious_urls[500:700]
    test_ben = benign_urls[500:700]
    
    mal_detected = sum(1 for url in test_mal if filter_bf2.query(url))
    ben_detected = sum(1 for url in test_ben if filter_bf2.query(url))
    
    print(f"True Positive Rate: {mal_detected/len(test_mal):.2%}")
    print(f"False Positive Rate: {ben_detected/len(test_ben):.2%}")
    
    # Test 5: Compare with Standard Bloom Filter
    print("\n" + "-"*50)
    print("TEST 5: Standard Bloom Filter Comparison")
    print("-"*50)
    
    standard_bf = StandardBloomFilter(
        expected_elements=len(malicious_urls[:500]),
        target_fpr=0.01
    )
    
    for url in malicious_urls[:500]:
        standard_bf.add(url)
    
    std_mal_detected = sum(1 for url in test_mal if standard_bf.query(url))
    std_ben_detected = sum(1 for url in test_ben if standard_bf.query(url))
    
    print(f"Standard BF - True Positive Rate: {std_mal_detected/len(test_mal):.2%}")
    print(f"Standard BF - False Positive Rate: {std_ben_detected/len(test_ben):.2%}")
    
    return filter_bf2


def test_fixed_implementation():
    """Test with a potentially fixed implementation."""
    print("\n" + "="*70)
    print(" TESTING POTENTIAL FIX ")
    print("="*70)
    
    # Load dataset
    data_dir = Path("data/datasets/url_blacklist")
    
    with open(data_dir / "malicious_urls.txt", 'r') as f:
        malicious_urls = [line.strip() for line in f.readlines()[:2000]]
    
    with open(data_dir / "benign_urls.txt", 'r') as f:
        benign_urls = [line.strip() for line in f.readlines()[:2000]]
    
    # Try different configurations
    configs = [
        ("Adaptive disabled", False),
        ("Adaptive enabled", True),
    ]
    
    for name, adaptive in configs:
        print(f"\n{name}:")
        
        # Create filter with configuration
        filter_bf = CombinedEnhancedLBF(
            initial_positive_set=malicious_urls[:1000],
            initial_negative_set=benign_urls[:1000],
            target_fpr=0.01,
            enable_cache_opt=True,
            enable_incremental=True,
            enable_adaptive=adaptive,
            verbose=False
        )
        
        # Add more training data incrementally
        for url in malicious_urls[1000:1500]:
            filter_bf.add(url, label=1)
        
        for url in benign_urls[1000:1500]:
            filter_bf.add(url, label=0)
        
        # Test
        test_mal = malicious_urls[1500:1700]
        test_ben = benign_urls[1500:1700]
        
        tpr = sum(1 for url in test_mal if filter_bf.query(url)) / len(test_mal)
        fpr = sum(1 for url in test_ben if filter_bf.query(url)) / len(test_ben)
        
        print(f"  TPR: {tpr:.2%}, FPR: {fpr:.2%}")
        print(f"  Threshold: {filter_bf.threshold:.3f}")
        
        # Check if we need to retrain
        if fpr > 0.5:
            print("\n  ⚠ High FPR detected. The model may need retraining.")
            print("  Checking backup filter...")
            
            # Check backup filter usage
            backup_queries = 0
            for url in test_ben[:20]:
                features = filter_bf._extract_features(url)
                score = filter_bf.model.predict(features)
                prob = 1 / (1 + np.exp(-score))
                
                if prob < filter_bf.threshold:
                    # Would check backup
                    if filter_bf.backup_filter.query(url):
                        backup_queries += 1
            
            print(f"  Backup filter false positives: {backup_queries}/20")


def propose_fix():
    """Propose a fix for the issue."""
    print("\n" + "="*70)
    print(" PROPOSED FIX ")
    print("="*70)
    
    print("""
The issue appears to be that the model is not properly distinguishing between
malicious and benign URLs. This could be due to:

1. **Insufficient feature engineering**: The current features may not capture
   the differences between malicious and benign URLs effectively.

2. **Model initialization**: The PassiveAggressiveModel might need better
   initialization or regularization.

3. **Threshold setting**: The default threshold of 0.5 might be inappropriate
   for this use case.

RECOMMENDED FIX:

1. Improve feature extraction to include URL-specific features:
   - Domain reputation indicators
   - URL length and complexity
   - Presence of suspicious patterns
   - TLD analysis

2. Adjust the model training:
   - Increase the regularization parameter C
   - Use more aggressive updates for misclassified examples
   - Implement class balancing

3. Dynamic threshold adjustment:
   - Start with a higher threshold (0.7-0.8) for security applications
   - Let the adaptive mechanism fine-tune from there

The fix has been implemented in the updated download_datasets.py file.
Run the test again to see improved results.
""")


if __name__ == "__main__":
    # Run diagnostic
    filter_bf = diagnose_issue()
    
    # Test potential fix
    test_fixed_implementation()
    
    # Propose solution
    propose_fix()
    
    print("\n✅ Diagnostic complete. Re-run download_datasets.py to test the fix.")