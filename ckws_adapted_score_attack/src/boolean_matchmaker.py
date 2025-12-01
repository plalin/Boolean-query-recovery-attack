"""
Boolean Score Attack Matchmaker

This module extends the conjunctive score attack to support boolean queries
(both conjunctive/AND and disjunctive/OR operations).

The BooleanScoreAttack class inherits from ConjunctiveScoreAttack and reuses
all its functionality since the co-occurrence matrix computation and matching
logic remains the same for boolean queries.
"""

import colorlog

from typing import Dict, List, Optional
import tensorflow as tf

from ckws_adapted_score_attack.src.conjunctive_matchmaker import ConjunctiveScoreAttack

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")


class BooleanScoreAttack(ConjunctiveScoreAttack):
    """
    Score attack for boolean queries (conjunctive + disjunctive).
    
    This class extends ConjunctiveScoreAttack to handle boolean keyword combinations
    that include both:
    - Conjunctive keywords (AND): A & B
    - Disjunctive keywords (OR): A | B
    
    The keywords are distinguished by their separators:
    - & for conjunctive (AND)
    - | for disjunctive (OR)
    
    The underlying attack algorithm remains the same as ConjunctiveScoreAttack,
    since boolean queries are treated as an extended keyword vocabulary.
    """

    def __init__(
        self,
        keyword_occ_array: tf.Tensor,
        keyword_sorted_voc: tf.Tensor,
        trapdoor_occ_array: tf.Tensor,
        trapdoor_sorted_voc: Optional[List[str]],
        known_queries: Dict[str, str],
        norm_ord=2,  # L2 (Euclidean norm),
        **kwargs,
    ):
        """
        Initialize the Boolean Score Attack matchmaker.

        Arguments:
            keyword_occ_array {tf.Tensor} -- Boolean keyword occurrence array 
                                            (rows: combinations, columns: similar documents)
                                            Includes both conjunctive and disjunctive keywords
            trapdoor_occ_array {tf.Tensor} -- Trapdoor occurrence array
                                            (rows: combinations, columns: stored documents)
            keyword_sorted_voc {tf.Tensor} -- Boolean keyword vocabulary with separators
                                            (& for AND, | for OR)
            trapdoor_sorted_voc {List[str]} -- Trapdoor vocabulary (hashed)
            known_queries {Dict[str, str]} -- Known query mappings (trapdoor -> keyword)
            norm_ord {int} -- Order of the norm used by the matchmaker (default: 2)
        
        Note:
            The keyword vocabulary should use different separators:
            - "A&B" for conjunctive (A AND B)
            - "A|B" for disjunctive (A OR B)
        """
        # Call parent constructor - all the logic is the same!
        super().__init__(
            keyword_occ_array=keyword_occ_array,
            keyword_sorted_voc=keyword_sorted_voc,
            trapdoor_occ_array=trapdoor_occ_array,
            trapdoor_sorted_voc=trapdoor_sorted_voc,
            known_queries=known_queries,
            norm_ord=norm_ord,
            **kwargs,
        )
        
        logger.info("Initialized BooleanScoreAttack (supports both conjunctive & disjunctive queries)")
        logger.debug(f"Boolean keyword vocabulary size: {len(keyword_sorted_voc)}")
        logger.debug(f"Trapdoor vocabulary size: {len(trapdoor_sorted_voc)}")

    def __str__(self):
        return "BooleanScoreAttack"
    
    def is_conjunctive(self, keyword: str) -> bool:
        """Check if a keyword is conjunctive (AND)."""
        return "&" in keyword
    
    def is_disjunctive(self, keyword: str) -> bool:
        """Check if a keyword is disjunctive (OR)."""
        return "|" in keyword
