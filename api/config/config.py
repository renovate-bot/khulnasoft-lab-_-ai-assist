import os


class Config:
    def __init__(self):
        """
        This class contains all the config options, most of them are derived from environment variables.
        Why a Python class? Because it allows for more advanced functionality and native support of the correct
        types.
        """
        self.triton_host = os.environ.get("TRITON_HOST", "triton")
        self.triton_port = os.environ.get("TRITON_PORT", 8001)
        self.triton_verbose = self._str_to_bool(os.environ.get("TRITON_VERBOSITY", False))
        self.docs_url = os.environ.get("DOCS_URL", None)
        self.openapi_url = os.environ.get("OPENAPI_URL", None)
        self.redoc_url = os.environ.get("REDOC_URL", None)
        self.bypass: bool = self._str_to_bool(os.environ.get("BYPASS_EXTERNAL_AUTH", False))
        self.api_host = "0.0.0.0"
        self.api_port: int = 5000
        self.gitlab_api_base_url = os.environ.get("GITLAB_API_URL", "https://gitlab.com/api/v4/")
        self.use_local_cache = self._str_to_bool(os.environ.get("USE_LOCAL_CACHE", True))

        self.uvicorn_logger = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": '%(levelprefix)s %(asctime)s :: %(client_addr)s - "%(request_line)s" %(status_code)s',
                    "use_colors": True
                },
            },
            "handlers": {
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn.access": {
                    "handlers": ["access"],
                    # "level": "INFO",
                    "propagate": False
                },
            },
        }

    @staticmethod
    def _str_to_bool(env_var):
        """
        Age old problem with env vars in Python, False becomes a str "False" which can not be compared against a bool,
        this function converts strings into booleans.
        :param env_var:
        :return:
        """
        if isinstance(env_var, bool):
            return env_var
        elif env_var.lower() == "true":
            return True
        elif env_var.lower() == "false":
            return False
        else:
            raise EnvironmentError(f"Config value is invalid: {env_var}")
