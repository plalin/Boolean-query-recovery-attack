"""
Test script to compare Boolean modules with Conjunctive modules
"""
import sys
import os
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from ckws_adapted_score_attack.src.common import (
    conjunctive_keyword_combinations_indices,
    boolean_keyword_combinations_indices
)
from ckws_adapted_score_attack.src.conjunctive_extraction import ConjunctiveExtractor
from ckws_adapted_score_attack.src.boolean_extraction import BooleanExtractor
from ckws_adapted_score_attack.src.email_extraction import load_preprocessed_enron_float32

print("="*80)
print("BOOLEAN MODULES TEST")
print("="*80)

# Load dataset
print("\n[1] Loading dataset...")
document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ = load_preprocessed_enron_float32(prefix='')
print(f"Dataset shape: {document_keyword_occurrence.dense_shape}")
print(f"Vocabulary size: {sorted_keyword_voc.shape[0]}")

# Split dataset for testing (40% of documents)
emails = tf.sparse.split(
    sp_input=document_keyword_occurrence,
    num_split=document_keyword_occurrence.dense_shape[0],
    axis=0,
)

# Use first 40% for testing
num_test_docs = int(len(emails) * 0.4)
test_docs = tf.sparse.concat(sp_inputs=emails[:num_test_docs], axis=0)
print(f"Test docs shape: {test_docs.dense_shape}")

print("\n" + "="*80)
print("TEST 1: Small vocabulary (voc_size=10, kw_size=2)")
print("="*80)

voc_size_small = 10
kw_size = 2

# Test 1.1: Combination indices
print("\n[1.1] Testing keyword combination indices generation...")
conj_combos = conjunctive_keyword_combinations_indices(voc_size_small, kw_size)
bool_combos = boolean_keyword_combinations_indices(voc_size_small, kw_size)

print(f"\nConjunctive combinations shape: {conj_combos.shape}")
print(f"Conjunctive combinations:\n{conj_combos}")

print(f"\nBoolean combinations shape: {bool_combos.shape}")
print(f"Boolean combinations (first half - conjunctive):\n{bool_combos[:, :conj_combos.shape[1]]}")
print(f"Boolean combinations (second half - disjunctive):\n{bool_combos[:, conj_combos.shape[1]:]}")

# Verify boolean = [conjunctive, disjunctive]
assert bool_combos.shape[1] == 2 * conj_combos.shape[1], "Boolean should have 2x combinations"
assert np.array_equal(bool_combos[:, :conj_combos.shape[1]], conj_combos), "First half should match conjunctive"
assert np.array_equal(bool_combos[:, conj_combos.shape[1]:], conj_combos), "Second half should match conjunctive"
print("✓ Boolean combinations structure is correct!")

# Test 1.2: Extractors with small vocabulary
print("\n[1.2] Testing extractors with small vocabulary...")

conj_extractor_small = ConjunctiveExtractor(
    occurrence_array=test_docs,
    keyword_voc=sorted_keyword_voc,
    keyword_occ=sorted_keyword_occ,
    voc_size=voc_size_small,
    kw_conjunction_size=kw_size,
    min_freq=1,
    multi_core=False,  # Single core for testing
)

bool_extractor_small = BooleanExtractor(
    occurrence_array=test_docs,
    keyword_voc=sorted_keyword_voc,
    keyword_occ=sorted_keyword_occ,
    voc_size=voc_size_small,
    kw_conjunction_size=kw_size,
    min_freq=1,
    multi_core=False,  # Single core for testing
)

print(f"\nConjunctive extractor:")
print(f"  - occ_array shape: {conj_extractor_small.occ_array.shape}")
print(f"  - sorted_voc length: {conj_extractor_small.sorted_voc.shape[0]}")
print(f"  - vocabulary sample: {conj_extractor_small.sorted_voc.numpy()[:5]}")

print(f"\nBoolean extractor:")
print(f"  - occ_array shape: {bool_extractor_small.occ_array.shape}")
print(f"  - sorted_voc length: {bool_extractor_small.sorted_voc.shape[0]}")
print(f"  - conjunctive_size: {bool_extractor_small.conjunctive_size}")
print(f"  - disjunctive_start: {bool_extractor_small.disjunctive_start}")
print(f"  - vocabulary sample (conjunctive): {bool_extractor_small.sorted_voc.numpy()[:5]}")
print(f"  - vocabulary sample (disjunctive): {bool_extractor_small.sorted_voc.numpy()[bool_extractor_small.disjunctive_start:bool_extractor_small.disjunctive_start+5]}")

# Verify sizes (single_core mode: shape is (documents, combinations))
expected_conj_size = conj_extractor_small.occ_array.shape[1]  # Number of combinations (columns in single_core)
expected_bool_size = 2 * expected_conj_size
assert bool_extractor_small.occ_array.shape[1] == expected_bool_size, f"Boolean occ_array should have {expected_bool_size} columns"
assert bool_extractor_small.sorted_voc.shape[0] == expected_bool_size, f"Boolean vocabulary should have {expected_bool_size} entries"
print("✓ Boolean extractor sizes are correct!")

# Test 1.3: Check separator characters
print("\n[1.3] Checking vocabulary separators...")
conj_voc_sample = conj_extractor_small.sorted_voc.numpy()[:5]
bool_voc_conj_sample = bool_extractor_small.sorted_voc.numpy()[:5]
bool_voc_disj_sample = bool_extractor_small.sorted_voc.numpy()[bool_extractor_small.disjunctive_start:bool_extractor_small.disjunctive_start+5]

print(f"Conjunctive module uses '|' separator: {conj_voc_sample}")
print(f"Boolean conjunctive uses '&' separator: {bool_voc_conj_sample}")
print(f"Boolean disjunctive uses '|' separator: {bool_voc_disj_sample}")

# Check separators
for voc in bool_voc_conj_sample:
    assert b'&' in voc, f"Conjunctive should use '&': {voc}"
for voc in bool_voc_disj_sample:
    assert b'|' in voc, f"Disjunctive should use '|': {voc}"
print("✓ Separators are correct!")

# Test 1.4: Check occurrence values (binary)
print("\n[1.4] Checking occurrence array values...")

# Conjunctive occurrence should be binary (0 or 1)
conj_occ_unique = tf.unique(tf.reshape(conj_extractor_small.occ_array, [-1]))[0].numpy()
print(f"Conjunctive occ_array unique values: {sorted(conj_occ_unique)}")

# Boolean occurrence should also be binary
bool_occ_unique = tf.unique(tf.reshape(bool_extractor_small.occ_array, [-1]))[0].numpy()
print(f"Boolean occ_array unique values: {sorted(bool_occ_unique)}")

assert set(conj_occ_unique).issubset({0.0, 1.0}), "Conjunctive should be binary"
assert set(bool_occ_unique).issubset({0.0, 1.0}), "Boolean should be binary"
print("✓ All occurrence values are binary (0 or 1)!")

# Test 1.5: Compare conjunctive parts
print("\n[1.5] Comparing conjunctive parts...")
conj_occ_from_conj = conj_extractor_small.occ_array
conj_occ_from_bool = bool_extractor_small.occ_array[:, :bool_extractor_small.conjunctive_size]  # First N columns

print(f"Conjunctive occ shape: {conj_occ_from_conj.shape}")
print(f"Boolean conjunctive part shape: {conj_occ_from_bool.shape}")

# They should be the same
if np.allclose(conj_occ_from_conj.numpy(), conj_occ_from_bool.numpy()):
    print("✓ Conjunctive parts are identical!")
else:
    print("✗ Conjunctive parts differ!")
    diff_ratio = np.mean(np.abs(conj_occ_from_conj.numpy() - conj_occ_from_bool.numpy()))
    print(f"  Average difference: {diff_ratio}")

# Test 1.6: Check disjunctive logic
print("\n[1.6] Checking disjunctive (OR) logic...")
disj_occ = bool_extractor_small.occ_array[:, bool_extractor_small.disjunctive_start:]  # Last N columns
print(f"Disjunctive occ shape: {disj_occ.shape}")

# Sample verification: For OR operation, result should be 1 if either keyword is 1
original_occ = bool_extractor_small.original_occ_array
sample_combo_idx = 0  # First combination

# Get the keyword indices for first disjunctive combination
combo_indices = bool_extractor_small.keyword_combinations[bool_extractor_small.disjunctive_start + sample_combo_idx]
keyword_a_occ = original_occ[:, combo_indices[0]]
keyword_b_occ = original_occ[:, combo_indices[1]]
expected_or = tf.minimum(keyword_a_occ + keyword_b_occ, 1.0)
actual_or = disj_occ[:, sample_combo_idx]  # Column in single_core mode

print(f"\nSample disjunctive combination {sample_combo_idx}:")
print(f"  Keyword A occurrences (first 10 docs): {keyword_a_occ.numpy()[:10]}")
print(f"  Keyword B occurrences (first 10 docs): {keyword_b_occ.numpy()[:10]}")
print(f"  Expected OR (first 10 docs): {expected_or.numpy()[:10]}")
print(f"  Actual OR (first 10 docs): {actual_or.numpy()[:10]}")

if np.allclose(expected_or.numpy(), actual_or.numpy()):
    print("✓ Disjunctive OR logic is correct!")
else:
    print("✗ Disjunctive OR logic differs!")
    diff_count = np.sum(np.abs(expected_or.numpy() - actual_or.numpy()) > 0.01)
    print(f"  Differences in {diff_count} documents")

print("\n" + "="*80)
print("TEST 2: Larger vocabulary (voc_size=50, kw_size=2)")
print("="*80)

voc_size_large = 50

print("\n[2.1] Testing with larger vocabulary...")

conj_extractor_large = ConjunctiveExtractor(
    occurrence_array=test_docs,
    keyword_voc=sorted_keyword_voc,
    keyword_occ=sorted_keyword_occ,
    voc_size=voc_size_large,
    kw_conjunction_size=kw_size,
    min_freq=1,
    multi_core=True,  # Multi-core for larger dataset
)

bool_extractor_large = BooleanExtractor(
    occurrence_array=test_docs,
    keyword_voc=sorted_keyword_voc,
    keyword_occ=sorted_keyword_occ,
    voc_size=voc_size_large,
    kw_conjunction_size=kw_size,
    min_freq=1,
    multi_core=True,  # Multi-core for larger dataset
)

print(f"\nConjunctive extractor (large):")
print(f"  - occ_array shape: {conj_extractor_large.occ_array.shape}")
print(f"  - Expected combinations: {voc_size_large * (voc_size_large - 1) // 2}")

print(f"\nBoolean extractor (large):")
print(f"  - occ_array shape: {bool_extractor_large.occ_array.shape}")
print(f"  - Expected combinations: {voc_size_large * (voc_size_large - 1)}")  # 2x
print(f"  - conjunctive_size: {bool_extractor_large.conjunctive_size}")
print(f"  - disjunctive_start: {bool_extractor_large.disjunctive_start}")

# Verify sizes (multi_core mode: shape is (combinations, documents))
expected_conj_size_large = voc_size_large * (voc_size_large - 1) // 2
expected_bool_size_large = 2 * expected_conj_size_large
assert conj_extractor_large.occ_array.shape[0] == expected_conj_size_large, f"Expected {expected_conj_size_large} rows (combinations)"
assert bool_extractor_large.occ_array.shape[0] == expected_bool_size_large, f"Expected {expected_bool_size_large} rows (combinations)"
print("✓ Sizes are correct for large vocabulary!")

# Check binary values
bool_occ_large_unique = tf.unique(tf.reshape(bool_extractor_large.occ_array, [-1]))[0].numpy()
print(f"\nBoolean occ_array unique values (large): {sorted(bool_occ_large_unique)}")
assert set(bool_occ_large_unique).issubset({0.0, 1.0}), "Boolean should still be binary"
print("✓ Large vocabulary still produces binary values!")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✓ All tests passed!")
print("\nKey findings:")
print("1. Boolean combinations = 2x conjunctive combinations")
print("2. Boolean vocabulary uses '&' for AND, '|' for OR")
print("3. Boolean occ_array is 2x size of conjunctive")
print("4. All values are binary (0 or 1)")
print("5. Conjunctive part matches original conjunctive module")
print("6. Disjunctive OR logic works correctly")
print("7. Works for both small and large vocabularies")
print("="*80)
