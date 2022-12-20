import uvicorn

from fastapi import FastAPI

from codesuggestions.api import http
from codesuggestions.api import middleware

app = FastAPI(
    title="GitLab Code Suggestions",
    description="GitLab Code Suggestions API to serve code completion predictions",
    middleware=[
        middleware.MiddlewareAuthentication(),
        middleware.MiddlewareLogRequest()
    ],
)

app.include_router(http.router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5052)
