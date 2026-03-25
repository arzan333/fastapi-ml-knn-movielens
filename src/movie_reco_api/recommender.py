from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ModelNotReadyError(RuntimeError):
    pass


class MovieNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class RecommenderArtifacts:
    movies: pd.DataFrame
    vectorizer: TfidfVectorizer
    knn: NearestNeighbors


def load_artifacts(artifacts_dir: Path) -> RecommenderArtifacts:
    movies_path = artifacts_dir / "movies.parquet"
    vec_path = artifacts_dir / "tfidf.joblib"
    knn_path = artifacts_dir / "knn.joblib"

    if not (movies_path.exists() and vec_path.exists() and knn_path.exists()):
        raise ModelNotReadyError(
            "Model artifacts not found. Run: uv run python scripts/train_knn.py"
        )

    movies = pd.read_parquet(movies_path)
    vectorizer = joblib.load(vec_path)
    knn = joblib.load(knn_path)

    return RecommenderArtifacts(movies=movies, vectorizer=vectorizer, knn=knn)


def recommend_by_title(art: RecommenderArtifacts, title: str, k: int) -> list[dict]:
    movies = art.movies
    if title not in set(movies["title"]):
        raise MovieNotFoundError(
            "Movie title not found. Use /movies/search to find the exact title."
        )

    idx = int(movies.index[movies["title"] == title][0])

    X = art.vectorizer.transform(movies["genre_doc"].values)
    dists, inds = art.knn.kneighbors(X[idx], n_neighbors=min(k + 1, len(movies)))

    out: list[dict] = []
    for movie_idx in inds[0]:
        if movie_idx == idx:
            continue
        rec_title = str(movies.iloc[movie_idx]["title"])
        genre_doc = str(movies.iloc[movie_idx]["genre_doc"])
        out.append({"title": rec_title, "reason": f"Similar genres: {genre_doc}"})
        if len(out) >= k:
            break

    return out
