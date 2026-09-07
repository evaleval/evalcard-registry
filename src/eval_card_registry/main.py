from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import math
import json

from eval_card_registry.config import settings
from eval_card_registry.store.hf_store import get_store, QUERY_TABLE_NAMES
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.services.log_writer import ResolveLogWriter
from eval_card_registry.api.routes_resolve import router as resolve_router
from eval_card_registry.api.routes_entities import router as entities_router
from eval_card_registry.api.routes_aliases import router as aliases_router
from eval_card_registry.api.routes_orgs import router as orgs_router
from eval_card_registry.api.routes_health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    if settings.read_only:
        store.load(tables=QUERY_TABLE_NAMES)
    else:
        store.load()

    # Singleton ResolutionService — avoids rebuilding AliasStore per request
    app.state.resolution_service = ResolutionService(store)

    # Resolve log writer
    log_writer = ResolveLogWriter(settings.hf_log_bucket)
    app.state.log_writer = log_writer
    log_writer.start(settings.log_flush_interval_seconds)

    yield

    await log_writer.stop()


class RegistryJSONResponse(JSONResponse):
    """JSON with a wire form for non-finite floats.

    Metric bounds use `.inf` / `-.inf` for "unbounded on that side" (see the
    seed/metrics.yaml header), and an infinite float is not JSON-compliant:
    Starlette's JSONResponse raises on one (allow_nan=False), and a bare
    json.dumps would emit the invalid literal `Infinity`. This renders them
    as the strings "Infinity" / "-Infinity", the form every_eval_ever's
    schema uses, and NaN as null. Registered as the app's default response
    class AND on the validation-error handler, since a client can put an
    infinity in a request body (`1e400` parses to inf) and the 422 echoes
    the offending input back.
    """

    @staticmethod
    def _finite(value):
        if isinstance(value, float):
            if math.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            if math.isnan(value):
                return None
            return value
        if isinstance(value, dict):
            return {k: RegistryJSONResponse._finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RegistryJSONResponse._finite(v) for v in value]
        return value

    def render(self, content) -> bytes:
        return json.dumps(
            self._finite(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


async def _validation_error(request: Request, exc: RequestValidationError):
    return RegistryJSONResponse(
        {"detail": jsonable_encoder(exc.errors())}, status_code=422
    )


app = FastAPI(
    title="eval-card-registry",
    description="Entity resolution registry for EEE evaluation data.",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=RegistryJSONResponse,
)
app.add_exception_handler(RequestValidationError, _validation_error)

PREFIX = "/api/v1"

app.include_router(resolve_router, prefix=PREFIX)
app.include_router(entities_router, prefix=PREFIX)
app.include_router(orgs_router, prefix=PREFIX)
app.include_router(aliases_router, prefix=PREFIX)
app.include_router(health_router, prefix=PREFIX)
