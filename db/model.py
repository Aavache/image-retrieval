from pydantic import BaseModel


class SearchParam(BaseModel):
    metric_type: str = "L2"
    nprobe: int = 10
    limit: int = 10