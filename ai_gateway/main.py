import uvicorn
from ai_gateway.app import get_config


def run_app():
    config = get_config()
    uvicorn.run(
        "ai_gateway.app:get_app",
        host=config.fastapi.api_host,
        port=config.fastapi.api_port,
        log_config=config.fastapi.uvicorn_logger,
        forwarded_allow_ips="*",
        reload=config.fastapi.reload,
    )


if __name__ == "__main__":
    run_app()
