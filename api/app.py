import logging
import hashlib

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from sse_starlette.sse import EventSourceResponse

from models import OpenAIinput
from utils.codegen import CodeGenProxy
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


@app.post("/v1/completions")
async def completions(data: OpenAIinput, request: Request):
    data = data.dict()
    # Trim whitespace from the stop sequences
    # TODO: move this to VS Code or preprocessing
    data['stop'] = [i.strip() for i in data['stop']]
    user = hashlib.sha1(request.client.host.encode("utf-8")).hexdigest()
    print(f"{user} - Received request")

    if gitlab.user_is_allowed(token=request.headers.get('authorization', '')):
        try:
            content = codegen(data=data)
        except codegen.TokensExceedsMaximum as E:
            print(f"{user} - Tokens exceeded")
            raise HTTPException(
                status_code=400,
                detail=str(E),
            )

        if data.get("stream") is not None:
            print(f"{user} - Returning suggestions")
            return EventSourceResponse(
                content=content,
                status_code=200,
                media_type="text/event-stream"
            )
        else:
            print(f"{user} - Returning suggestions")
            return Response(
                status_code=200,
                content=content,
                media_type="application/json"
            )
    else:
        print(f"{user} - Failed to get suggestions 404")
        return Response(
            status_code=404,
            content="{}",
            media_type="application/json"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host=config.api_host, port=config.api_port)
