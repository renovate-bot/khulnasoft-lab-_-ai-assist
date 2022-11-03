import logging

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from models import OpenAIinput
from utils.codegen import CodeGenProxy
from utils.errors import FauxPilotException
from utils.gitlab import GitLab
from config.config import Config

config = Config()

logging.config.dictConfig(config.uvicorn_logger)

codegen = CodeGenProxy(
    host=config.triton_host,
    port=config.triton_port,
    verbose=config.triton_verbose
)

gitlab = GitLab(bypass=config.bypass, base_url=config.gitlab_api_base_url)

app = FastAPI(
    title="GitLab AI Assist",
    description="GitLab AI Assist API to serve code completion predictions",
    openapi_url=config.openapi_url,
    docs_url=config.docs_url,
    redoc_url=config.redoc_url,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)


@app.exception_handler(FauxPilotException)
async def fauxpilot_handler(exc: FauxPilotException):
    return JSONResponse(
        status_code=400,
        content=exc.json()
    )


@app.post("/v1/completions")
async def completions(data: OpenAIinput, request: Request):
    data = data.dict()
    if gitlab.user_is_allowed(token=request.headers.get('authorization', '')):
        try:
            content = codegen(data=data)
        except codegen.TokensExceedsMaximum as E:
            raise FauxPilotException(
                message=str(E),
                error_type="invalid_request_error",
                param=None,
                code=None,
            )

        if data.get("stream") is not None:
            return EventSourceResponse(
                content=content,
                status_code=200,
                media_type="text/event-stream"
            )
        else:
            return Response(
                status_code=200,
                content=content,
                media_type="application/json"
            )
    else:
        return Response(
            status_code=404,
            content="{}",
            media_type="application/json"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host=config.api_host, port=config.api_port)
