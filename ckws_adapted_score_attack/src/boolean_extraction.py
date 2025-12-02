import hashlib
import multiprocessing
import random
from typing import Tuple, Dict, List

import tensorflow as tf
import numpy as np

import colorlog

from ckws_adapted_score_attack.src.common import boolean_keyword_combinations_indices
from ckws_adapted_score_attack.src.config import DATA_TYPE
from ckws_adapted_score_attack.src.conjunctive_extraction import sparse_gather

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")

MAX_CPUS = 50  # multiprocessing.cpu_count() if multiprocessing.cpu_count() <= 128 else 128


class BooleanExtractor:
    """
    Class to extract occurrence array for boolean queries (conjunctive + disjunctive)
    from preprocessed dataset.
    
    Boolean queries include both:
    - Conjunctive (AND): A & B
    - Disjunctive (OR): A | B
    
    Keyword combinations are ordered as: [conjunctive, disjunctive]
    
    Artificial keyword combinations is optional, but can be used to save time when running 
    a lot of experiments.
        NB: The keyword combinations won't change if you run many examples on just different 
        dataset splits.
    """

    def __init__(
            self,
            occurrence_array: tf.SparseTensor,
            keyword_voc: tf.Tensor,
            keyword_occ: tf.Tensor,
            voc_size: int = 100,
            kw_conjunction_size: int = 1,
            min_freq: int = 1,
            precalculated_boolean_keyword_combinations_indices: np.ndarray = None,
            multi_core: bool = True,
    ):
        assert voc_size <= occurrence_array.shape[1], "Vocabulary size should be less or equal than total keywords"

        self.kw_conjunction_size = kw_conjunction_size
        self.voc_size = voc_size
        self.min_freq = min_freq

        # Calculate global occurrence per keyword
        column_sums = tf.sparse.reduce_sum(sp_input=occurrence_array, axis=0)

        logger.debug(f"len column sums: {column_sums.shape}")

        # Sort the global keywords and pick 'voc_size' most common
        self.original_keyword_indices = tf.cast(tf.argsort(column_sums, direction='DESCENDING'), dtype=tf.int64)
        self.original_keyword_indices = self.original_keyword_indices[:self.voc_size]

        # Store keywords tensor{'string'}
        self.keywords = tf.gather(keyword_voc, self.original_keyword_indices)

        # Column sums, since occurrence will be different each experiment
        self.occurrences = tf.gather(column_sums, self.original_keyword_indices)

        logger.info(f"Vocabulary size: {self.keywords.shape[0]}")

        # Reduce occurrence array size to match with keyword 'voc_size'
        logger.debug(f"Gather occurrence matrix columns for vocabulary")
        _occ_array = sparse_gather(occurrence_array, self.original_keyword_indices, axis=1)
        _occ_array = tf.sparse.reorder(_occ_array)
        self.original_occ_array = tf.sparse.to_dense(_occ_array)

        """
        original_occ_array: num_documents × voc_size dense matrix (single keywords)
        occ_array: num_documents × (2 * num_combinations) matrix (boolean keywords)
        """

        if self.kw_conjunction_size > 1:
            # Make keyword combinations for boolean queries
            logger.debug(f"Generate boolean keyword combinations (conjunctive + disjunctive)")

            if precalculated_boolean_keyword_combinations_indices is None:
                boolean_combinations = boolean_keyword_combinations_indices(
                    num_keywords=self.voc_size,
                    kw_conjunction_size=self.kw_conjunction_size
                )
            else:
                boolean_combinations = precalculated_boolean_keyword_combinations_indices

            # Calculate split point between conjunctive and disjunctive
            num_total_combinations = boolean_combinations.shape[1]
            self.conjunctive_size = num_total_combinations // 2
            self.disjunctive_start = self.conjunctive_size

            # Split into conjunctive and disjunctive parts
            conjunctive_combinations = boolean_combinations[:, :self.conjunctive_size].T
            disjunctive_combinations = boolean_combinations[:, self.conjunctive_size:].T

            self.keyword_combinations = boolean_combinations.T

            # Get keyword indices for vocabulary
            conj_voc_indices = tf.sort(tf.gather(self.original_keyword_indices, conjunctive_combinations))
            disj_voc_indices = tf.sort(tf.gather(self.original_keyword_indices, disjunctive_combinations))

            # Create vocabulary with different separators for AND/OR
            conj_voc = tf.strings.reduce_join(
                tf.gather(keyword_voc, conj_voc_indices), 
                separator="&",  # & for conjunctive (AND)
                axis=-1
            )
            disj_voc = tf.strings.reduce_join(
                tf.gather(keyword_voc, disj_voc_indices), 
                separator="|",  # | for disjunctive (OR)
                axis=-1
            )

            # Concatenate vocabularies: [conjunctive, disjunctive]
            self.sorted_voc = tf.concat([conj_voc, disj_voc], axis=0)

            # Build occurrence arrays using TensorArray for memory efficiency
            logger.info(f"Computing boolean occurrence array, kw_size={self.kw_conjunction_size}")
            logger.info("Forcing CPU execution to avoid GPU OOM")

            # Force CPU execution to avoid GPU memory issues with large tensors
            with tf.device("/device:CPU:0"):
                if multi_core:
                    # Combine both conjunctive and disjunctive into one TensorArray
                    self.occ_array = increase_occ_array_boolean_combined_multi_core(
                        original_occ=self.original_occ_array,
                        conjunctive_combinations=conjunctive_combinations,
                        disjunctive_combinations=disjunctive_combinations
                    )
                else:
                    self.occ_array = increase_occ_array_boolean_combined_single_core(
                        original_occ=self.original_occ_array,
                        conjunctive_combinations=conjunctive_combinations,
                        disjunctive_combinations=disjunctive_combinations
                    )

        else:
            self.conjunctive_size = self.voc_size
            self.disjunctive_start = self.voc_size
            self.sorted_voc = self.keywords
            self.occ_array = self.original_occ_array

        logger.debug(f"Boolean occ array shape: {self.occ_array.shape}")
        logger.debug(f"Conjunctive size: {self.conjunctive_size}, Disjunctive start: {self.disjunctive_start}")

    def get_sorted_voc(self) -> tf.Tensor:
        """Returns the sorted vocabulary without the occurrences.

        Returns:
            tf.Tensor<tf.string> -- Vocabulary (concatenated) keywords list with separators
                                    & for conjunctive (AND), | for disjunctive (OR)
        """
        return self.sorted_voc

    def is_conjunctive(self, idx: int) -> bool:
        """Check if keyword at index is conjunctive (AND)."""
        return idx < self.conjunctive_size

    def is_disjunctive(self, idx: int) -> bool:
        """Check if keyword at index is disjunctive (OR)."""
        return idx >= self.disjunctive_start


@tf.function
def increase_occ_array_boolean_combined_multi_core(
        original_occ: tf.Tensor,
        conjunctive_combinations: tf.Tensor,
        disjunctive_combinations: tf.Tensor
) -> tf.Tensor:
    """
    Compute boolean occurrence array using TensorArray for memory efficiency.
    Processes all combinations (conjunctive + disjunctive) in a single pass.
    
    Args:
        original_occ: Original occurrence array (documents × keywords)
        conjunctive_combinations: Conjunctive (AND) keyword combination indices
        disjunctive_combinations: Disjunctive (OR) keyword combination indices
    
    Returns:
        Occurrence array for all boolean combinations (combinations × documents)
    """
    total_size = tf.shape(conjunctive_combinations)[0] + tf.shape(disjunctive_combinations)[0]
    
    column_vectors = tf.TensorArray(
        dtype=DATA_TYPE,
        size=total_size,
        element_shape=tf.TensorShape(dims=[original_occ.shape[0]]),
        dynamic_size=False,
        name="init_tensorarray_boolean_occ_array"
    )

    conj_size = tf.shape(conjunctive_combinations)[0]

    def tf_while_body(idx, results):
        # Determine if we're processing conjunctive or disjunctive
        is_conjunctive = tf.less(idx, conj_size)
        
        # Select the appropriate combination
        combination = tf.cond(
            is_conjunctive,
            lambda: conjunctive_combinations[idx],
            lambda: disjunctive_combinations[idx - conj_size]
        )
        
        keyword_vectors = tf.gather(
            original_occ,
            combination,
            axis=1,
            name="gather_occ_array_boolean_combined"
        )

        # Apply operation based on type
        result_vector = tf.cond(
            is_conjunctive,
            lambda: tf.math.reduce_prod(keyword_vectors, axis=1, name="reduce_prod_conjunctive"),
            lambda: tf.minimum(tf.reduce_sum(keyword_vectors, axis=1, name="reduce_sum_disjunctive"), 1.0, name="min_disjunctive_binary")
        )

        return tf.add(idx, 1, name="add_idx"), \
               results.write(idx, result_vector, name="write_column_vector")

    i = tf.constant(0)
    tf_while_cond = lambda loop_idx, x: tf.less(loop_idx, total_size)

    idx, result = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[i, column_vectors],
        parallel_iterations=MAX_CPUS,
        name="while_loop_boolean_keyword_array_combined"
    )

    return result.stack(name="stack_boolean_keyword_vectors")


@tf.function
def increase_occ_array_boolean_combined_single_core(
        original_occ: tf.Tensor,
        conjunctive_combinations: tf.Tensor,
        disjunctive_combinations: tf.Tensor
) -> tf.Tensor:
    """
    Compute boolean occurrence array using TensorArray (single-core version).
    
    Args:
        original_occ: Original occurrence array (documents × keywords)
        conjunctive_combinations: Conjunctive (AND) keyword combination indices
        disjunctive_combinations: Disjunctive (OR) keyword combination indices
    
    Returns:
        Occurrence array for all boolean combinations (documents × combinations)
    """
    total_size = tf.shape(conjunctive_combinations)[0] + tf.shape(disjunctive_combinations)[0]
    
    column_vectors = tf.TensorArray(
        dtype=DATA_TYPE,
        size=total_size,
        dynamic_size=False,
        name="init_tensorarray_boolean_occ_array_single"
    )

    conj_size = tf.shape(conjunctive_combinations)[0]
    idx = tf.constant(0)
    
    # Process conjunctive combinations
    for combination in conjunctive_combinations:
        keyword_vectors = tf.gather(
            original_occ, 
            combination, 
            axis=1, 
            name="gather_occ_array_conjunctive"
        )
        result = tf.math.reduce_prod(keyword_vectors, axis=1, name="reduce_prod_conjunctive")
        column_vectors = column_vectors.write(idx, result)
        idx = tf.add(idx, 1)
    
    # Process disjunctive combinations
    for combination in disjunctive_combinations:
        keyword_vectors = tf.gather(
            original_occ, 
            combination, 
            axis=1, 
            name="gather_occ_array_disjunctive"
        )
        added = tf.reduce_sum(keyword_vectors, axis=1, name="reduce_sum_disjunctive")
        result = tf.minimum(added, 1.0, name="min_disjunctive_binary")
        column_vectors = column_vectors.write(idx, result)
        idx = tf.add(idx, 1)

    # Stack and transpose for single_core (documents × combinations)
    stacked = column_vectors.stack(name="stack_boolean_keyword_vectors_single")
    return tf.transpose(stacked, name="transpose_boolean_occ_single_core")


@tf.function
def increase_occ_array_boolean_multi_core(
        original_occ: tf.Tensor,
        keyword_combinations: tf.Tensor,
        operation: str = "conjunctive"
) -> tf.Tensor:
    """
    Compute occurrence array for boolean keyword combinations (multi-core).
    
    Args:
        original_occ: Original occurrence array (documents × keywords)
        keyword_combinations: Keyword combination indices
        operation: "conjunctive" for AND (multiplication), "disjunctive" for OR (addition + min)
    
    Returns:
        Occurrence array for keyword combinations
    """
    column_vectors = tf.TensorArray(
        dtype=DATA_TYPE,
        size=tf.shape(keyword_combinations)[0],
        element_shape=tf.TensorShape(dims=[original_occ.shape[0]]),
        dynamic_size=False,
        name="init_tensorarray_boolean_occ_array"
    )

    def tf_while_body(idx, results):
        keyword_vectors = tf.gather(
            original_occ,
            keyword_combinations[idx],
            axis=1,
            name="gather_occ_array_boolean"
        )

        if operation == "conjunctive":
            # AND operation: multiplication (binary: 1*1=1, 1*0=0, 0*0=0)
            multiplied = tf.math.reduce_prod(keyword_vectors, axis=1, name="reduce_prod_conjunctive")
        else:  # disjunctive
            # OR operation: addition + min with 1 to keep binary (binary: min(1+1,1)=1, min(1+0,1)=1, min(0+0,1)=0)
            added = tf.reduce_sum(keyword_vectors, axis=1, name="reduce_sum_disjunctive")
            multiplied = tf.minimum(added, 1.0, name="min_disjunctive_binary")

        return tf.add(idx, 1, name="add_idx"), \
               results.write(idx, multiplied, name="write_column_vector")

    i = tf.constant(0)
    tf_while_cond = lambda loop_idx, x: tf.less(loop_idx, tf.shape(keyword_combinations)[0])

    idx, result = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[i, column_vectors],
        parallel_iterations=MAX_CPUS,
        name="while_loop_boolean_keyword_array"
    )

    return result.stack(name="stack_boolean_keyword_vectors")


@tf.function
def increase_occ_array_boolean_single_core(
        original_occ: tf.Tensor,
        keyword_combinations: tf.Tensor,
        operation: str = "conjunctive",
        core: int = -1,
) -> tf.Tensor:
    """
    Compute occurrence array for boolean keyword combinations (single-core).
    
    Args:
        original_occ: Original occurrence array (documents × keywords)
        keyword_combinations: Keyword combination indices
        operation: "conjunctive" for AND (multiplication), "disjunctive" for OR (addition + min)
        core: Core identifier (for logging, not used)
    
    Returns:
        Occurrence array for keyword combinations
    """
    column_vectors = tf.TensorArray(
        dtype=DATA_TYPE,
        size=tf.shape(keyword_combinations)[0],
        dynamic_size=False,
        name="init_tensorarray_boolean_occ_array"
    )

    logger.info(f"Number of keyword combinations: {keyword_combinations.shape[0]}, operation: {operation}")
    idx = tf.constant(0, name="initialize_idx_0_constant")
    
    for keyword_combination in keyword_combinations:
        keyword_vectors = tf.gather(
            original_occ, 
            keyword_combination, 
            axis=1, 
            name="gather_occ_array_single_core"
        )

        if operation == "conjunctive":
            # AND operation: multiplication
            result = tf.math.reduce_prod(keyword_vectors, axis=1, name="reduce_prod_conjunctive")
        else:  # disjunctive
            # OR operation: addition + min with 1
            added = tf.reduce_sum(keyword_vectors, axis=1, name="reduce_sum_disjunctive")
            result = tf.minimum(added, 1.0, name="min_disjunctive_binary")

        column_vectors = column_vectors.write(idx, result, name="write_column_vector")
        idx = tf.add(idx, 1, name="increase_idx")

    # Stack creates (num_combinations, num_documents), transpose to (num_documents, num_combinations)
    return tf.transpose(column_vectors.stack(name="stack_boolean_keyword_vectors"), name="transpose_boolean_occ_array")
