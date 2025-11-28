import csv
import gc
import hashlib
import multiprocessing
import random

# PYTHONPATH에 프로젝트 루트 디렉토리 추가 (디버깅용)
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
# 디버깅 안할 시 주석부터 사이 영역 제거


import colorlog
import numpy as np
from scipy.special import comb
import tensorflow as tf
import time

from ckws_adapted_score_attack.src.conjunctive_extraction import ConjunctiveExtractor, generate_trapdoors, generate_known_queries
from ckws_adapted_score_attack.src.conjunctive_matchmaker import ConjunctiveScoreAttack
from ckws_adapted_score_attack.src.conjunctive_query_generator import ConjunctiveQueryResultExtractor
from ckws_adapted_score_attack.src.common import KeywordExtractor, conjunctive_keyword_combinations_indices
from ckws_adapted_score_attack.src.common import generate_known_queries as old_generate_known_queries
from ckws_adapted_score_attack.src.email_extraction import (
    split_df,
    extract_sent_mail_contents,
    extract_apache_ml,
    load_enron, load_preprocessed_dataset,
    load_preprocessed_enron, load_preprocessed_enron_float32,
)
from ckws_adapted_score_attack.src.query_generator import (
    QueryResultExtractor
)
from ckws_adapted_score_attack.src.matchmaker import ScoreAttack
from ckws_adapted_score_attack.src.trapdoor_matchmaker import memory_usage


MAX_CPUS = 64    # multiprocessing.cpu_count() if multiprocessing.cpu_count() <= 128 else 128

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")
NB_REP = 1
kw_conjunction_size = 2
start_time = time.time()

def boolean_naive_base_results(result_file=f"{kw_conjunction_size}-kws_boolean_naive_base_attack-{start_time}.csv"):
    with tf.device("/device:CPU:0"):
        # voc_size_possibilities = [200]
        voc_size_possibilities = [150]
        known_queries_possibilities = [60]
        experiment_params = [
            (i, j)
            for i in voc_size_possibilities         # Voc size
            for j in known_queries_possibilities    # known queries
            for _k in range(NB_REP)
        ]

        with open(result_file, "w", newline="") as csv_file:
            fieldnames = [
                "Nb similar docs",
                "Nb server docs",
                "Similar/Server voc size",
                "Nb queries seen",
                "Nb queries known",
                "Refinement acc",
            ]
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()

            document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ = load_preprocessed_enron_float32(prefix='')
            
            """
            document_keyword_occurrence: 30109 * 62976 "sparse" matrix
            entries: # of occurrences of each keyword in each document

            keyword_sorted_voc: 키워드들을 빈도순으로 정렬한 1D array, 각 인덱스 값이 해당 위치의 키워드 문자열
                
            sorted_keyword_voc[0] = "the"      # 가장 자주 나오는 키워드
            sorted_keyword_voc[1] = "and"      # 두 번째로 자주 나오는 키워드
            sorted_keyword_voc[2] = "is"       # 세 번째로 자주 나오는 키워드
            ... 
            sorted_keyword_voc[62975] = "xyz"  # 가장 적게 나오는 키워드

            sorted_keyword_occ: 키워드들의 빈도수가 엔트리인 1D array, 각 인덱스 값이 해당 위치의 키워드 빈도수
            sorted_keyword_occ[0] = 1000000  # "the" 키워드의 빈도수
            sorted_keyword_occ[1] = 500000   # "and" 키워드의 빈도수
            sorted_keyword_occ[2] = 300000   # "is" 키워드의 빈도수
            ...
            sorted_keyword_occ[62975] = 1    # "xyz" 키워드의 빈도수            
            """

            # Split to N sparse tensors: one sparse tensor per email
            emails = tf.sparse.split(
                sp_input=document_keyword_occurrence,
                num_split=document_keyword_occurrence.dense_shape[0],
                axis=0,
            )

            logger.debug(f"Number of emails: {len(emails)}")

            modulus = len(known_queries_possibilities) * NB_REP

            keyword_combinations = None

            for (i, (voc_size, nb_known_queries)) in enumerate(experiment_params):
                logger.info(f"Experiment {i+1} out of {len(experiment_params)}")
                memory_usage()

                if i % modulus == 0:
                    logger.debug(f"Generate keyword combinations: nCr({voc_size}, {kw_conjunction_size})")
                    keyword_combinations = conjunctive_keyword_combinations_indices(
                        num_keywords=voc_size, kw_conjunction_size=kw_conjunction_size)

                # Split similar and real dataset
                email_ids = list(range(document_keyword_occurrence.dense_shape[0]))
                random.shuffle(email_ids)

                similar_doc_ids, stored_doc_ids = email_ids[:int(len(emails) * 0.4)], email_ids[int(len(emails) * 0.4):]
                logger.debug(f"Generated random email ids")

                similar_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in similar_doc_ids], axis=0)
                logger.debug(f"Similar docs shape: {similar_docs.dense_shape}")

                stored_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in stored_doc_ids], axis=0)
                logger.debug(f"Stored docs shape: {stored_docs.dense_shape}")

                """
                similar_docs: 30109*0.4 × 62976 sparse matrix
                stored_docs: 30109*0.6 × 62976 sparse matrix
                """

                # Extract keywords from similar dataset
                memory_usage()
                similar_extractor = ConjunctiveExtractor(
                    occurrence_array=similar_docs,
                    keyword_voc=sorted_keyword_voc,
                    keyword_occ=sorted_keyword_occ,
                    voc_size=voc_size,
                    kw_conjunction_size=kw_conjunction_size,
                    min_freq=1,
                    precalculated_artificial_keyword_combinations_indices=keyword_combinations,
                    multi_core=True,
                )
                nb_similar_docs = similar_extractor.occ_array.shape[1]
                memory_usage()

                """
                similar_extractor.occ_array: 각 문서에서 키워드의 조합이 발생한 횟수가 엔트리인 행렬 (conjunctive keyword combinations * nb_similar_docs)

                occ_array = O라고 할 때, O^T * O 연산은 conjunctive_matchmaker.py에 포함됨

                TODO: occ_array의 열에 해당하는 엔트리가 boolean (conjunctive + disjunctive) 가 되도록 수정
                keyword_combinations를 boolean_combinations로 수정
                """

                # Extract keywords from real dataset
                real_extractor = ConjunctiveQueryResultExtractor(
                    stored_docs,
                    sorted_keyword_voc,
                    sorted_keyword_occ,
                    voc_size,
                    kw_conjunction_size,
                    1,
                    keyword_combinations,
                    True,
                )
                nb_server_docs = real_extractor.occ_array.shape[1]
                memory_usage()

                """
                real_extractor.occ_array: ConjunctiveExtractor.occ_array와 동일한 형태의 행렬 (conjunctive keyword combinations * nb_server_docs)
                """ 

                # Queries = 15% of artificial kws
                queryset_size = int(comb(voc_size, kw_conjunction_size, exact=True) * 0.15)

                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                """
                query_voc = ["alice|bob", "email|meeting", "project|report", "time|schedule"] 와 같이 임의로 선정된 키워드 조합의 리스트

                query_array = [
                [15,  0,  7,  0,  3],   # query 0 ("alice|bob")의 각 문서 발생 횟수
                [ 0,  8,  0,  2,  0],   # query 1 ("email|meeting")
                [ 6,  0,  0,  4,  1],   # query 2 ("project|report")
                [ 0,  5,  3,  0,  0],   # query 3 ("time|schedule")
                ]
                (shape: [4, 5])

                *hide_nb_files=True이면 모든 query_voc의 쿼리들에 대해 값이 0인 문서들을 뺌
                """

                # remove from memory not needed
                del real_extractor

                known_queries = generate_known_queries(
                    similar_wordlist=similar_extractor.get_sorted_voc().numpy(),
                    stored_wordlist=query_voc.numpy(),
                    nb_queries=nb_known_queries,
                )
                memory_usage()

                """
                similar dataset과 stored dataset의 conjunctive keyword pool의 교집합에서
                nb_known_queries개의 키워드를 랜덤으로 선정

                known_queries = {
                "alice|bob": "alice|bob",
                "email|meeting": "email|meeting",
                ... (총 60개)
                }
                """

                td_voc, known_queries, eval_dict = generate_trapdoors(
                    query_voc=query_voc.numpy(),
                    known_queries=known_queries,
                )

                """
                td_voc: query_voc의 각 키워드를 해시한 리스트.
                
                query_voc = [
                "alice|bob",
                "email|meeting",
                "project|report",
                "time|schedule",
                ...
                ]

                td_voc = [
                "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",  # SHA1("alice|bob")
                "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1",  # SHA1("email|meeting")
                "c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2",  # SHA1("project|report")
                "d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3",  # SHA1("time|schedule")
                ...
                ]
                """

                """
                # 가정: 전체 쿼리 100개, 알려진 쿼리 10개

                # 1. td_voc: 모든 trapdoor 토큰 (리스트)
                td_voc = [
                    "hash1", "hash2", "hash3", ..., "hash100"
                ]
                (length: 100)

                # 2. known_queries: 알려진 trapdoor만 (딕셔너리)
                known_queries = {
                    "hash1": "keyword1",   # 공격자가 이미 알고 있음
                    "hash5": "keyword5",
                    "hash23": "keyword23",
                    ...
                }
                (length: 10)

                # 3. eval_dict: 모든 trapdoor의 정답 (딕셔너리)
                eval_dict = {
                    "hash1": "keyword1",
                    "hash2": "keyword2",
                    "hash3": "keyword3",
                    ...,
                    "hash100": "keyword100"
                }
                (length: 100)
                """
                # known_queries := Keys: Trapdoor tokens; Values: Keywords
                memory_usage()

                matchmaker = ConjunctiveScoreAttack(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc().numpy(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                )
                memory_usage()

                # Difference known queries and fake queries are the trapdoors we want to recover
                td_list = list(set(eval_dict.keys()).difference(matchmaker.known_queries.keys()))

                refinement_speed = int(0.05 * queryset_size)
                logger.info(f"Refinement speed: {refinement_speed}")

                # Prediction with refinement, but without clustering
                results = matchmaker.tf_predict_with_refinement(
                    td_list,
                    cluster_max_size=1,
                    ref_speed=refinement_speed
                )
                memory_usage()
                refinement_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                del similar_extractor

                writer.writerow(
                    {
                        "Nb similar docs": nb_similar_docs,
                        "Nb server docs": nb_server_docs,
                        "Similar/Server voc size": voc_size,
                        "Nb queries seen": queryset_size,
                        "Nb queries known": nb_known_queries,
                        "Refinement acc": refinement_accuracy,
                    }
                )

                # Flush, such that intermediate results don't get lost if something bad happens
                csv_file.flush()

                # Garbage collection to prevent high memory usage (tends to increase overtime)
                gc.collect()