import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from inverted_index_gcp import *
import hashlib
from collections import defaultdict
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq

def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)
#Extract calculations from buket
bucket_name = '318964772'

file_path = 'calculations/docid_title_pairs.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
doc_title_pairs = pickle.loads(contents)

file_path = 'calculations/docid_body_length.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
doc_lengths = pickle.loads(contents)


bucket_name = '318964772'
file_path = 'calculations_title_index/docid_title_length.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
title_lengths = pickle.loads(contents)


bucket_name = '318964772'
file_path = 'page_rank.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
page_rank_dict = pickle.loads(contents)

bucket_name = '318964772'
file_path = 'page_views.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
page_views = pickle.loads(contents)

bucket_name = '318964772'
file_path = 'title_index/title_index.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
title_index_stem = pickle.loads(contents)

bucket_name = '318964772'
file_path = 'title_idf_dict_stem.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
title_idf_stem = pickle.loads(contents)


bucket_name = '318964772'
file_path = 'body_index_with_stem/body_index_with_stem.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
body_index_stem = pickle.loads(contents)

bucket_name = '318964772'
file_path = 'word_idf_dict_stem.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
idf_stem = pickle.loads(contents)


filter_func = lambda tok: tok not in all_stopwords
tokenize = lambda text: [token.group() for token in RE_WORD.finditer(text.lower()) if token not in all_stopwords]


################################################newest######################################################
def calculate_score_title_batch(words, title_index_stem, lock, title_idf_stem, title_lengths,th):
    """
        Calculate scores for documents based on the BM25 algorithm for a batch of words in the title index.

        Args:
            words (list): List of words in the query batch.
            title_index_stem: Instance of the title index.
            lock (threading.Lock): Thread lock to ensure safe access to shared resources.
            title_idf_stem (dict): IDF scores for words in the title index.
            title_lengths (dict): Lengths of documents in the title index.

        Returns:
            defaultdict: Dictionary containing document IDs as keys and their corresponding scores.
        """
    k1 = 1.5
    b = 0.75
    title_scores = defaultdict(float)
    for word in words:
        if title_idf_stem[word] < th:  # Skip words with low IDF values
            continue
        pl = title_index_stem.read_a_posting_list(base_dir="", w=word, bucket_name="318964772")

        for docid, tf in pl:
            score = title_idf_stem[word]*(tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (title_lengths[docid] / 2.6211557574449786)))
            with lock:
                title_scores[docid] += score
    return title_scores

def search_query_title_stem_bm25(query):
    """
       Search for documents based on the BM25 algorithm using the title index.

       Args:
           query (list): List of words in the query.

       Returns:
           list: List of tuples containing document IDs and their corresponding scores.
       """
    query_weights = {}
    if len(query) <=2:
        th = 1.6
    else:
        th = 2
    for word in query:
        if title_index_stem.df.get(word, -10) == -10:
            continue
        tf = query.count(word)
        weight = tf * title_idf_stem[word]
        query_weights[word] = weight

    title_scores = defaultdict(float)
    lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        # Split the query words into batches
        words_batches = [list(query_weights.keys())[i:i+4] for i in range(0, len(query_weights), 4)]
        futures = []
        for words_batch in words_batches:
            futures.append(executor.submit(calculate_score_title_batch, words_batch, title_index_stem, lock, title_idf_stem, title_lengths,th))
        for future in futures:
            title_scores.update(future.result())

    # Sort the results by score and return the top 100
    ret = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)[:400]
    return ret

################################################newest######################################################


################################################newest######################################################
def calculate_score_body_batch(words, body_index_stem, idf_stem, doc_lengths, lock, title_scores,th):
    """
      Calculate scores for documents based on the BM25 algorithm for a batch of words in the body index.

      Args:
          words (list): List of words in the query batch.
          body_index_stem: Instance of the body index.
          idf_stem (dict): IDF scores for words in the body index.
          doc_lengths (dict): Lengths of documents in the body index.
          lock (threading.Lock): Thread lock to ensure safe access to shared resources.
          title_scores (defaultdict): Dictionary containing document IDs as keys and their corresponding scores.

      Returns:
          defaultdict: Updated dictionary containing document IDs and their corresponding scores.
      """
    k1 = 1.5
    b = 0.75
    for word in words:
        if idf_stem[word] < th: #use for runtime efficient
            continue
        pl = body_index_stem.read_a_posting_list(base_dir="", w=word, bucket_name="318964772")
        for docid, tf in pl:
            score = idf_stem[word]*(tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_lengths[docid] / 431.1623426698441)))
            with lock:
                title_scores[docid] += score
    return title_scores

def search_query_body_stem_bm25(query):
    """
        Search for documents based on the BM25 algorithm using the body index.

        Args:
            query (list): List of words in the query.

        Returns:
            list: List of tuples containing document IDs and their corresponding scores.
        """


    query_weights = {}
    if len(query) >=6:
        th = 1.15
    else:
        th = 1.5
    for word in query:
        if body_index_stem.df.get(word, -10) == -10:
            continue

        tf = query.count(word)
        weight = tf * idf_stem[word]
        query_weights[word] = weight

    title_scores = defaultdict(float)
    lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        words_batches = [list(query_weights.keys())[i:i+4] for i in range(0, len(query_weights), 4)]
        futures = []
        for words_batch in words_batches:
            futures.append(executor.submit(calculate_score_body_batch, words_batch,body_index_stem, idf_stem, doc_lengths, lock, title_scores,th))
        for future in futures:
            title_scores.update(future.result())

    ret = sorted([(doc_id, score) for doc_id, score in title_scores.items()], key=lambda x: x[1], reverse=True)[:400]
    return ret

################################################newest######################################################


def calculate_title_scores_merge( title_results, page_rank_dict, title_weight):
    """
        Calculate combined scores for documents based on title search results, page rank, and page views.

        Args:
            title_results (list): List of tuples containing document IDs and their corresponding scores from the title search.
            page_rank_dict (dict): Dictionary containing page ranks for documents.
            title_weight (float): Weight assigned to title scores.

        Returns:
            defaultdict: Dictionary containing document IDs as keys and their combined scores as values.
        """
    title_scores = defaultdict(float)
    for doc_id, score in title_results:
        page_view_weight = page_views[doc_id]
        if page_views[doc_id] == 0:
            page_view_weight = 1

        title_scores[doc_id] += score * title_weight + 2*math.log(page_rank_dict[doc_id], 10) + 2*math.log(page_view_weight,10)
    return title_scores

def calculate_body_scores_merge( body_results, page_rank_dict, body_weight):
    """
       Calculate combined scores for documents based on body search results, page rank, and page views.

       Args:
           body_results (list): List of tuples containing document IDs and their corresponding scores from the body search.
           page_rank_dict (dict): Dictionary containing page ranks for documents.
           body_weight (float): Weight assigned to body scores.

       Returns:
           defaultdict: Dictionary containing document IDs as keys and their combined scores as values.
       """
    body_scores = defaultdict(float)
    for doc_id, score in body_results:
        page_view_weight = page_views[doc_id]
        if page_views[doc_id] == 0:
            page_view_weight = 1

        body_scores[doc_id] += score * body_weight + math.log(page_rank_dict[doc_id], 10) + math.log(page_view_weight,10)
    return body_scores

def search(query):
    """
        Search for documents based on the given query.

        Args:
            query (str): Input query.

        Returns:
            list: List of tuples containing document IDs and their corresponding titles, sorted by relevance score.
        """
    # Weights for different components
    title_weight = 0.6
    body_weight = 0.4
    query = tokenize(query.lower())
    stemmer = PorterStemmer()
    query = [stemmer.stem(token) for token in query]
    if len(query) <=2:
        body_weight = 0
        title_weight = 1
    if len(query) >=6:
        title_weight = 0.2
        body_weight =0.8
    # Get sorted results from individual functions
    title_results = search_query_title_stem_bm25(query)
    body_results = search_query_body_stem_bm25(query)
    # Calculate page ranks in parallel
    with ThreadPoolExecutor() as executor:
        title_scores_future = executor.submit(calculate_title_scores_merge, title_results, page_rank_dict, title_weight)
        body_scores_future = executor.submit(calculate_body_scores_merge,body_results, page_rank_dict, body_weight)

    title_scores = title_scores_future.result()
    body_scores = body_scores_future.result()

    # Combine results into a single dictionary (document ID as key, score as value)
    combined_scores = defaultdict(float)
    for doc_id, score in title_scores.items():
        combined_scores[doc_id] += score
    for doc_id, score in body_scores.items():
        combined_scores[doc_id] += score

    # Sort documents by combined score in descending order
    ret = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    ret = [(str(doc_id), doc_title_pairs[doc_id]) for doc_id, score in ret][:100]
    return ret