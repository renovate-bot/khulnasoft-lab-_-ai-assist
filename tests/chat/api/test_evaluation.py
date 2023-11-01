from fastapi import FastAPI

from ai_gateway.api.v1.chat import router


def test_non_availability():
    app = FastAPI()
    app.include_router(router)

    is_available = any([r.path.startswith("/evaluation") for r in app.routes])

    # We don't set F_CHAT_EVALUATION_API and ANTHROPIC_API_KEY by default,
    # so the evaluation endpoints need to be unavailable
    assert not is_available
