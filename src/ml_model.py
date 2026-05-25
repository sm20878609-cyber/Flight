"""
ml_model.py
-----------
Machine-learning core for the AI Paper Reading Assistant:
  - TF-IDF vectorisation
  - Cosine-similarity ranking
  - Keyword extraction (TF-IDF weights)
  - KMeans topic clustering
  - PCA 2-D projection
"""

import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


# ──────────────────────────────────────────────────────────────
# TF-IDF Vectorisation
# ──────────────────────────────────────────────────────────────

def build_tfidf_matrix(
    corpus: List[str],
    max_features: int = 5000,
    min_df: int = 1,
    max_df: float = 0.95,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit a TF-IDF vectoriser on the given corpus and return the document matrix.

    Parameters
    ----------
    corpus       : List[str] – one string per document
    max_features : int       – vocabulary size cap
    min_df       : int       – minimum document frequency for a term
    max_df       : float     – maximum document frequency (as fraction)

    Returns
    -------
    (matrix, vectorizer)
        matrix     : np.ndarray of shape (n_docs, n_features)
        vectorizer : fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        ngram_range=(1, 2),      # unigrams + bigrams for richer features
        sublinear_tf=True,       # log-normalise term frequency
    )
    matrix = vectorizer.fit_transform(corpus).toarray()
    return matrix, vectorizer


# ──────────────────────────────────────────────────────────────
# Query-to-Paper Cosine Similarity
# ──────────────────────────────────────────────────────────────

def rank_papers_by_similarity(
    paper_texts: List[str],
    user_query: str,
    max_features: int = 5000,
) -> Tuple[np.ndarray, TfidfVectorizer, np.ndarray]:
    """
    Compute cosine similarity between the user query and each paper.

    Parameters
    ----------
    paper_texts : List[str] – cleaned text of each paper
    user_query  : str       – user's research interest string
    max_features: int

    Returns
    -------
    (similarity_scores, vectorizer, tfidf_matrix)
        similarity_scores : np.ndarray of shape (n_papers,)
        vectorizer        : fitted TfidfVectorizer
        tfidf_matrix      : np.ndarray of shape (n_papers, n_features)
    """
    if not paper_texts:
        return np.array([]), None, np.array([])

    # Combine corpus: papers first, query last
    corpus = paper_texts + [user_query]

    min_df = 1 if len(paper_texts) == 1 else 2
    # Relax min_df when corpus is tiny
    if len(corpus) <= 3:
        min_df = 1

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.97,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_all = vectorizer.fit_transform(corpus).toarray()

    paper_vecs = tfidf_all[:-1]   # all rows except last
    query_vec  = tfidf_all[-1:]   # last row

    scores = cosine_similarity(query_vec, paper_vecs).flatten()
    return scores, vectorizer, paper_vecs


# ──────────────────────────────────────────────────────────────
# Keyword Extraction
# ──────────────────────────────────────────────────────────────

def extract_keywords(
    paper_text: str,
    vectorizer: TfidfVectorizer,
    top_k: int = 8,
) -> List[str]:
    """
    Extract the top-k keywords from a single paper using TF-IDF weights.

    Parameters
    ----------
    paper_text : str             – cleaned text of one paper
    vectorizer : TfidfVectorizer – already fitted on the full corpus
    top_k      : int             – number of keywords to return

    Returns
    -------
    List[str] – top-k keywords sorted by TF-IDF weight (descending)
    """
    if not paper_text or vectorizer is None:
        return []

    vec = vectorizer.transform([paper_text]).toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()

    # Pair terms with their weights and sort
    term_weights = sorted(
        zip(feature_names, vec), key=lambda x: x[1], reverse=True
    )
    keywords = [term for term, weight in term_weights if weight > 0]
    return keywords[:top_k]


def extract_keywords_standalone(
    paper_text: str,
    all_texts: List[str],
    top_k: int = 8,
) -> List[str]:
    """
    Fit a fresh TF-IDF vectoriser on `all_texts` and extract keywords
    for `paper_text`. Useful when the main vectoriser is query-inclusive.

    Parameters
    ----------
    paper_text : str
    all_texts  : List[str] – corpus (without query) for IDF computation
    top_k      : int

    Returns
    -------
    List[str]
    """
    if not paper_text or not all_texts:
        return []

    min_df = 1
    kw_vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=min_df,
        max_df=0.97,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        kw_vectorizer.fit(all_texts)
        return extract_keywords(paper_text, kw_vectorizer, top_k)
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
# Inter-Paper Cosine Similarity Matrix
# ──────────────────────────────────────────────────────────────

def compute_similarity_matrix(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all papers.

    Parameters
    ----------
    tfidf_matrix : np.ndarray – shape (n_papers, n_features)

    Returns
    -------
    np.ndarray – symmetric matrix of shape (n_papers, n_papers)
    """
    if tfidf_matrix is None or tfidf_matrix.size == 0:
        return np.array([[]])
    return cosine_similarity(tfidf_matrix)


# ──────────────────────────────────────────────────────────────
# KMeans Topic Clustering
# ──────────────────────────────────────────────────────────────

def cluster_papers(
    tfidf_matrix: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Cluster papers into `n_clusters` groups using KMeans.

    Returns None labels and a warning message if the data is too sparse
    (fewer papers than clusters, or all-zero vectors).

    Parameters
    ----------
    tfidf_matrix : np.ndarray – shape (n_papers, n_features)
    n_clusters   : int        – desired number of clusters
    random_state : int

    Returns
    -------
    (labels, warning_msg)
        labels      : np.ndarray of int, or None on failure
        warning_msg : str or None
    """
    if tfidf_matrix is None or tfidf_matrix.size == 0:
        return None, "No data for clustering."

    n_papers = tfidf_matrix.shape[0]

    if n_papers < 2:
        return None, "Clustering requires at least 2 papers. Skipping."

    if n_papers < n_clusters:
        n_clusters = n_papers
        msg = f"Fewer papers than requested clusters. Using {n_clusters} clusters."
    else:
        msg = None

    # Guard: check for zero-variance (all-zero) rows
    row_norms = np.linalg.norm(tfidf_matrix, axis=1)
    if np.all(row_norms == 0):
        return None, "All TF-IDF vectors are zero – cannot cluster."

    try:
        km = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        labels = km.fit_predict(tfidf_matrix)
        return labels, msg
    except Exception as exc:
        return None, f"KMeans failed: {exc}"


# ──────────────────────────────────────────────────────────────
# PCA 2-D Projection
# ──────────────────────────────────────────────────────────────

def compute_pca_2d(
    tfidf_matrix: np.ndarray,
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Project the TF-IDF matrix to 2 dimensions using PCA.

    Parameters
    ----------
    tfidf_matrix : np.ndarray – shape (n_papers, n_features)
    random_state : int

    Returns
    -------
    (coords_2d, warning_msg)
        coords_2d   : np.ndarray of shape (n_papers, 2), or None on failure
        warning_msg : str or None
    """
    if tfidf_matrix is None or tfidf_matrix.size == 0:
        return None, "No data for PCA."

    n_papers, n_features = tfidf_matrix.shape

    if n_papers < 2:
        return None, "PCA requires at least 2 papers. Skipping."

    n_components = min(2, n_papers, n_features)
    if n_components < 2:
        return None, "Not enough dimensions for 2-D PCA."

    try:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(tfidf_matrix)
        return coords, None
    except Exception as exc:
        return None, f"PCA failed: {exc}"


# ──────────────────────────────────────────────────────────────
# Build Result DataFrame
# ──────────────────────────────────────────────────────────────

def build_results_dataframe(
    filenames:   List[str],
    titles:      List[str],
    summaries:   List[str],
    keywords:    List[List[str]],
    scores:      np.ndarray,
    labels:      Optional[np.ndarray],
) -> pd.DataFrame:
    """
    Assemble all per-paper results into a ranked Pandas DataFrame.

    Parameters
    ----------
    filenames : List[str]
    titles    : List[str]
    summaries : List[str]
    keywords  : List[List[str]] – each element is a list of keyword strings
    scores    : np.ndarray      – cosine similarity scores
    labels    : np.ndarray or None – KMeans cluster labels

    Returns
    -------
    pd.DataFrame sorted by Similarity Score descending, with a Rank column.
    """
    n = len(filenames)
    cluster_col = (
        [f"Cluster {int(l)}" for l in labels]
        if labels is not None and len(labels) == n
        else ["—"] * n
    )

    df = pd.DataFrame(
        {
            "File Name":        filenames,
            "Extracted Title":  titles,
            "Short Summary":    [s[:300] + "…" if len(s) > 300 else s for s in summaries],
            "Keywords":         [", ".join(kws) if kws else "—" for kws in keywords],
            "Similarity Score": np.round(scores, 4) if len(scores) == n else [0.0] * n,
            "Cluster":          cluster_col,
        }
    )

    df.sort_values("Similarity Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "Rank", range(1, n + 1))
    return df
