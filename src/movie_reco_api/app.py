from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from movie_reco_api.recommender import (
    ModelNotReadyError,
    MovieNotFoundError,
    load_artifacts,
    recommend_by_title,
)
from movie_reco_api.schemas import RecommendRequest, RecommendResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="Movie Recommender API (KNN + MovieLens 100k)",
        version="0.1.0",
        description="Input one movie title and get KNN-based recommendations.",
    )

    artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
    state = {"art": None}

    @app.on_event("startup")
    def _startup() -> None:
        try:
            state["art"] = load_artifacts(artifacts_dir)
        except ModelNotReadyError:
            state["art"] = None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_ready": state["art"] is not None}

    @app.get("/movies/search")
    def search_movies(q: str = Query(..., min_length=1, max_length=50)) -> dict:
        if state["art"] is None:
            raise HTTPException(status_code=503, detail="Model not ready. Train it first.")
        movies = state["art"].movies
        hits = movies[movies["title"].str.contains(q, case=False, na=False)].head(20)
        return {"query": q, "results": hits["title"].tolist()}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        if state["art"] is None:
            raise HTTPException(status_code=503, detail="Model not ready. Train it first.")
        try:
            recs = recommend_by_title(state["art"], req.movie_title, req.k)
        except MovieNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return RecommendResponse(
            input_title=req.movie_title,
            recommendations=recs,
        )

    @app.exception_handler(Exception)
    def unhandled_exception_handler(_, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": "Internal error", "type": type(exc).__name__})

    return app


app = create_app()

