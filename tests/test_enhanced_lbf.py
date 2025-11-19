"""
Unit tests for Enhanced Learned Bloom Filter components
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_lbf.combined import CombinedEnhancedLBF
from src.enhanced_lbf.incremental import IncrementalLBF


def test_combined_no_false_negatives():
    positives = [f"malware_site_{i}.com/bad.php" for i in range(50)]
    negatives = [f"https://www.google.com/search?q={i}" for i in range(50)]

    lbf = CombinedEnhancedLBF(
        initial_positive_set=positives,
        initial_negative_set=negatives,
        target_fpr=0.01,
        enable_cache_opt=True,
        enable_incremental=True,
        enable_adaptive=False,
        verbose=False,
    )

    for item in positives:
        assert lbf.query(item) is True


def test_combined_backup_policy_respects_threshold():
    lbf = CombinedEnhancedLBF(
        target_fpr=0.01,
        enable_cache_opt=True,
        enable_incremental=True,
        enable_adaptive=False,
        verbose=False,
    )

    # Force routing to backup at insert
    lbf.threshold = 0.99
    before = lbf.positive_backup.count
    lbf.add("http://example.com/neutral_item", label=1)
    after = lbf.positive_backup.count
    assert after == before + 1

    # Now force not adding to backup
    lbf.threshold = 0.01
    before2 = lbf.positive_backup.count
    lbf.add("http://example.com/neutral_item2", label=1)
    after2 = lbf.positive_backup.count
    assert after2 == before2


def test_combined_cache_short_circuit():
    lbf = CombinedEnhancedLBF(
        target_fpr=0.01,
        enable_cache_opt=True,
        enable_incremental=True,
        enable_adaptive=False,
        verbose=False,
    )

    # Ensure item goes to backup and sets cache bit
    lbf.threshold = 0.9999
    lbf.add("http://cached-item.test/path", label=1)

    hits_before = lbf.cache_hits
    # Query a DIFFERENT item that definitely isn't in the set
    # and hopefully doesn't collide in the cache block (probabilistic, but likely with 1 item)
    # We need to ensure it routes to backup (low score) but is caught by cache
    lbf.threshold = 0.9999
    result = lbf.query("http://definitely-not-there.com/safe")
    
    # If the cache optimization works, it should return False WITHOUT checking backup
    # (assuming no collision in the small cache block)
    if result is False:
         # If result is False, it could be because of cache or backup.
         # But we want to check if cache_hits increased.
         # Note: There's a small chance of collision. If collision, it's a miss.
         # So we might need a loop to find a non-colliding item if we want to be robust.
         pass

    # Let's try to find a non-colliding item
    for i in range(100):
        item = f"http://not-there-{i}.com"
        hits_before = lbf.cache_hits
        result = lbf.query(item)
        if result is False and lbf.cache_hits == hits_before + 1:
            # Found a cache hit (short circuit)
            return

    # If we get here, we failed to find a short-circuitable item (unlikely with empty filter)
    # OR the cache logic is broken.
    assert False, "Cache optimization did not short-circuit any queries"


def test_incremental_features_length():
    inc = IncrementalLBF(verbose=False)
    features = inc._extract_features("https://example.com/malware.php")
    assert len(features) == 20
    assert len(inc.model.weights) == 20
    # Smoke test add/query
    inc.add("https://example.com/malware.php", label=1)
    assert inc.query("https://example.com/malware.php") in [True, False]
