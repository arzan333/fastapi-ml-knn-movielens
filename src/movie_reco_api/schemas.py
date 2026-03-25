from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    movie_title: str = Field(..., min_length=1, max_length=200, description="Exact movie title from MovieLens.")
    k: int = Field(3, ge=1, le=10, description="How many recommendations to return (max 10).")


class MovieRecommendation(BaseModel):
    title: str
    reason: str


class RecommendResponse(BaseModel):
    input_title: str
    recommendations: list[MovieRecommendation]
