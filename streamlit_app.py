from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

GENRES = [
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


@st.cache_data(show_spinner=False)
def load_top10_per_genre() -> dict[str, list[str]]:
    root = Path(__file__).resolve().parent
    movies_path = root / "artifacts" / "movies.parquet"
    ratings_path = root / "data" / "ml-100k" / "ml-100k" / "u.data"

    if not movies_path.exists() or not ratings_path.exists():
        return {g: [] for g in GENRES}

    movies = pd.read_parquet(movies_path)
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "ts"],
    )

    counts = ratings.groupby("movie_id").size().reset_index(name="rating_count")
    df = movies.merge(counts, on="movie_id", how="left")
    df["rating_count"] = df["rating_count"].fillna(0).astype(int)

    df["genres"] = df["genre_doc"].astype(str).str.split(" ")
    df = df.explode("genres").rename(columns={"genres": "genre"})
    df["genre"] = df["genre"].fillna("unknown")

    out: dict[str, list[str]] = {}
    for g in GENRES:
        top = (
            df[df["genre"] == g]
            .sort_values(["rating_count", "title"], ascending=[False, True])
            .head(10)["title"]
            .tolist()
        )
        out[g] = top

    return out


st.set_page_config(page_title="Movie Recommender (KNN)", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommender (KNN + MovieLens 100k)")
st.caption("Pick a movie (genre-wise shortlist) OR type any exact MovieLens title → get recommendations via FastAPI.")

with st.sidebar:
    st.header("API")
    api = st.text_input("Base URL", value=API_BASE)
    st.markdown("---")
    st.subheader("Quick Pick (Top 10 per Genre)")
    top10 = load_top10_per_genre()

    genre = st.selectbox("Genre", GENRES, index=GENRES.index("Comedy") if "Comedy" in GENRES else 0)
    choices = top10.get(genre, [])

    picked = st.selectbox("Top 10 movies", ["(choose one)"] + choices) if choices else "(no data found)"
    if not choices:
        st.warning("Top-10 list not available. Run download + training first.")

try:
    h = requests.get(f"{api}/health", timeout=5).json()
    if not h.get("model_ready"):
        st.warning("API is up, but model is NOT ready. Run training first (scripts/train_knn.py).")
    else:
        st.success("API is up and model is ready ✅")
except Exception:
    st.error("API is not reachable. Start FastAPI first (uvicorn on port 8000).")
    st.stop()

st.markdown("### 1) Search for a movie title (optional)")
q = st.text_input("Search keyword", placeholder="e.g., Toy Story")
if st.button("Search"):
    r = requests.get(f"{api}/movies/search", params={"q": q}, timeout=10)
    if r.status_code == 200:
        results = r.json()["results"]
        if results:
            st.write("Matches:")
            st.code("\n".join(results))
        else:
            st.info("No matches found.")
    else:
        st.error(r.json().get("detail", "Search failed"))

st.markdown("### 2) Get recommendations")
default_title = "" if picked in ("(choose one)", "(no data found)") else picked
movie_title = st.text_input(
    "Exact MovieLens title (you can type any title here)",
    value=default_title,
    placeholder="e.g., Toy Story (1995)",
)

k = st.slider("How many recommendations?", min_value=1, max_value=10, value=3)

if st.button("Recommend 🎯"):
    payload = {"movie_title": movie_title, "k": k}
    r = requests.post(f"{api}/recommend", json=payload, timeout=15)

    if r.status_code == 200:
        data = r.json()
        st.subheader(f"Recommendations for: {data['input_title']}")
        for i, rec in enumerate(data["recommendations"], start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {rec['title']}**")
                st.caption(rec["reason"])
    else:
        st.error(r.json().get("detail", "Recommendation failed"))
