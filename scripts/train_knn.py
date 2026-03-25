from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def load_movies(u_item_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        u_item_path,
        sep="|",
        encoding="latin-1",
        header=None,
    )

    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "FilmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western",
    ]

    df = df.rename(
        columns={
            0: "movie_id",
            1: "title",
            2: "release_date",
            3: "video_release_date",
            4: "imdb_url",
        }
    )

    for i, g in enumerate(genre_cols):
        df[g] = df[5 + i].astype(int)

    def genre_doc(row: pd.Series) -> str:
        tokens: list[str] = []
        for g in genre_cols:
            if row[g] == 1:
                tokens.append(g)
        return " ".join(tokens) if tokens else "unknown"

    df["genre_doc"] = df.apply(genre_doc, axis=1)
    keep = ["movie_id", "title", "genre_doc"]
    return df[keep].copy()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    u_item = root / "data" / "ml-100k" / "ml-100k" / "u.item"
    if not u_item.exists():
        raise FileNotFoundError(f"Missing {u_item}. Run scripts/download_data.py first.")

    movies = load_movies(u_item)

    vectorizer = TfidfVectorizer(lowercase=False)
    X = vectorizer.fit_transform(movies["genre_doc"].values)

    knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
    knn.fit(X)

    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    movies_path = artifacts / "movies.parquet"
    vec_path = artifacts / "tfidf.joblib"
    knn_path = artifacts / "knn.joblib"

    movies.to_parquet(movies_path, index=False)
    joblib.dump(vectorizer, vec_path)
    joblib.dump(knn, knn_path)

    sample = "Toy Story (1995)"
    idx = int(np.where(movies["title"].values == sample)[0][0]) if sample in set(movies["title"]) else 0
    dists, inds = knn.kneighbors(X[idx], n_neighbors=4)
    recs = movies.iloc[inds[0]].copy()
    print("Sample input:", movies.iloc[idx]["title"])
    print("Top 3 recommendations:")
    print(recs.iloc[1:][["title", "genre_doc"]].to_string(index=False))

    print("\nSaved artifacts:")
    print(movies_path)
    print(vec_path)
    print(knn_path)


if __name__ == "__main__":
    main()
