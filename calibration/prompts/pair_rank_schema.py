from typing import List
from pydantic import BaseModel


class PairRankSchema(BaseModel):
    ranked_ratings: list
    likelihoods: list