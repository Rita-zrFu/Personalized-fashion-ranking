"""Six H&M retrieval sources fitted strictly before a recommendation cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

SOURCES = ["popularity", "segment", "recent", "cooccurrence", "als", "content"]


def age_segment(age):
    if pd.isna(age):
        return "unknown"
    return "<25" if age < 25 else "25-34" if age < 35 else "35-44" if age < 45 else "45+"


def implicit_als(matrix, factors=24, iterations=8, alpha=15., regularization=.1, seed=42):
    """Hu et al. confidence-weighted implicit ALS, using sparse exact updates.

    Minimize sum_ui c_ui (p_ui - x_u @ y_i)^2 + reg*(||X||²+||Y||²),
    with p_ui=1 for observed purchases and c_ui=1+alpha*r_ui.
    The missing-item contribution is retained through Y.T@Y / X.T@X.
    """
    rng = np.random.default_rng(seed)
    users = rng.normal(0, .01, (matrix.shape[0], factors))
    items = rng.normal(0, .01, (matrix.shape[1], factors))
    identity = np.eye(factors) * regularization

    def update(rows, other):
        gram = other.T @ other + identity
        result = np.zeros((rows.shape[0], factors))
        for row in range(rows.shape[0]):
            lo, hi = rows.indptr[row:row+2]
            idx, value = rows.indices[lo:hi], rows.data[lo:hi]
            if len(idx):
                selected = other[idx]
                extra = alpha * value
                system = gram + (selected.T * extra) @ selected
                result[row] = np.linalg.solve(system, selected.T @ (1 + extra))
        return result

    with threadpool_limits(limits=1):
        for iteration in range(iterations):
            users = update(matrix, items)
            items = update(matrix.T.tocsr(), users)
            print(f"  ALS iteration {iteration + 1}/{iterations}", flush=True)
    return users.astype("float32"), items.astype("float32")


def keep_top_neighbors(matrix, n=50):
    matrix = matrix.tocsr()
    data, indices, indptr = [], [], [0]
    for row in range(matrix.shape[0]):
        lo, hi = matrix.indptr[row:row+2]
        vals, cols = matrix.data[lo:hi], matrix.indices[lo:hi]
        valid = cols != row
        vals, cols = vals[valid], cols[valid]
        order = np.argsort(-vals, kind="stable")[:n]
        data.extend(vals[order]); indices.extend(cols[order]); indptr.append(len(data))
    return sparse.csr_matrix((data, indices, indptr), shape=matrix.shape, dtype="float32")


class Retrieval:
    def __init__(self, history, customers, articles, cutoff, config):
        self.history, self.cutoff, self.config = history, pd.Timestamp(cutoff), config
        self.users = sorted(history.customer_id.unique())
        self.items = np.asarray(sorted(history.article_id.unique()))
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.item_index = {i: j for j, i in enumerate(self.items)}
        self.ages = customers.set_index("customer_id").age.to_dict()
        self.segments = {u: age_segment(a) for u, a in self.ages.items()}
        rows = history.customer_id.map(self.user_index).to_numpy()
        cols = history.article_id.map(self.item_index).to_numpy()
        days = (self.cutoff - history.t_dat).dt.days.to_numpy()
        weights = np.exp(-days / 30).astype("float32")
        self.interactions = sparse.csr_matrix((weights, (rows, cols)),
                                               shape=(len(self.users), len(self.items)))
        binary = self.interactions.copy()
        binary.data[:] = 1
        counts = np.asarray(binary.sum(axis=0)).ravel()
        co = (binary.T @ binary).tocoo()
        co.data /= np.sqrt(counts[co.row] * counts[co.col])
        self.co = keep_top_neighbors(co, n=50)
        print(f"  Retrieval: {len(self.users):,} history users / {len(self.items):,} eligible items", flush=True)
        self.user_factors, self.item_factors = implicit_als(
            self.interactions, factors=config["als_factors"], iterations=config["als_iterations"], seed=config["seed"])
        meta = articles.set_index("article_id").loc[self.items]
        text = meta[["product_type_name", "product_group_name", "colour_group_name",
                     "department_name", "garment_group_name", "detail_desc"]].fillna("unknown").agg(" ".join, axis=1)
        self.tfidf = TfidfVectorizer(min_df=2, max_features=15000, ngram_range=(1, 2))
        tokens = self.tfidf.fit_transform(text)
        components = min(config["content_components"], tokens.shape[1] - 1, tokens.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=components, random_state=config["seed"])
        with threadpool_limits(limits=1):
            self.content = normalize(self.svd.fit_transform(tokens)).astype("float32")
            self.profiles = normalize(self.interactions @ self.content).astype("float32")
        recent = history[history.t_dat.ge(self.cutoff - pd.Timedelta(days=28))].copy()
        recent["weight"] = np.exp(-(self.cutoff - recent.t_dat).dt.days / 14)
        self.popularity = recent.groupby("article_id").weight.sum().reindex(self.items, fill_value=0).to_numpy()
        recent["segment"] = recent.customer_id.map(self.segments).fillna("unknown")
        segment_counts = recent.groupby(["segment", "article_id"]).weight.sum()
        self.segment_scores = {s: segment_counts.loc[s].reindex(self.items, fill_value=0).to_numpy()
                               for s in segment_counts.index.get_level_values(0).unique()}
        self.first_seen = history.groupby("article_id").t_dat.min().to_dict()

    def scores(self, user):
        idx = self.user_index.get(user)
        scores = {"popularity": self.popularity,
                  "segment": self.segment_scores.get(self.segments.get(user, "unknown"), self.popularity)}
        if idx is None:
            for source in ["recent", "cooccurrence", "als", "content"]:
                scores[source] = np.zeros(len(self.items), dtype="float32")
        else:
            row = self.interactions.getrow(idx)
            scores["recent"] = row.toarray().ravel()
            scores["cooccurrence"] = (row @ self.co).toarray().ravel()
            scores["als"] = self.item_factors @ self.user_factors[idx]
            scores["content"] = self.content @ self.profiles[idx]
        return scores

    def top_indices(self, values, n, require_positive=False):
        indices = np.arange(len(values))
        if require_positive:
            indices = indices[values > 0]
        if len(indices) > n:
            local = np.argpartition(-values[indices], n - 1)[:n]
            indices = indices[local]
        return indices[np.lexsort((self.items[indices], -values[indices]))][:n]

    def generate(self, users, omit_source=None):
        records, source_lists = [], {s: {} for s in SOURCES}
        n, cap = self.config["per_source"], self.config["max_candidates"]
        with threadpool_limits(limits=1):
            for user in sorted(users):
                scores = self.scores(user)
                ranks, reciprocal = {}, {}
                for source in SOURCES:
                    # A zero-history user cannot be retrieved by learned history sources.
                    if user not in self.user_index and source in ["recent", "cooccurrence", "als", "content"]:
                        chosen = np.array([], dtype=int)
                    else:
                        chosen = self.top_indices(scores[source], n,
                                                  source in ["popularity", "segment", "recent", "cooccurrence"])
                    source_lists[source][user] = self.items[chosen].tolist()
                    if source == omit_source:
                        continue
                    ranks[source] = {int(i): rank for rank, i in enumerate(chosen, 1)}
                    for i, rank in ranks[source].items():
                        reciprocal[i] = reciprocal.get(i, 0.) + 1 / (60 + rank)
                pool = sorted(reciprocal, key=lambda i: (-reciprocal[i], self.items[i]))[:cap]
                for i in pool:
                    row = dict(customer_id=user, article_id=self.items[i], retrieval_rrf=reciprocal[i])
                    for source in SOURCES:
                        row[f"score_{source}"] = float(scores[source][i])
                        row[f"from_{source}"] = int(i in ranks.get(source, {}))
                        row[f"rank_{source}"] = ranks.get(source, {}).get(i, n + 1)
                    records.append(row)
        return pd.DataFrame(records), source_lists

    def baseline_predictions(self, source_lists, users, k):
        """Fill a short personal list with past-only global popularity."""
        fallback = self.items[self.top_indices(self.popularity, k)].tolist()
        return {s: {u: list(dict.fromkeys(source_lists[s].get(u, []) + fallback))[:k]
                    for u in users} for s in SOURCES}
