"""HTTP entry point; health remains independent of model connectivity."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI

from app.config import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Missing model credentials are allowed; malformed settings fail at startup.
    get_settings()
    yield


app = FastAPI(title="MewHelp", lifespan=lifespan)


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str | bool]:
    return {"status": "ok", "model_configured": settings.model_configured}
