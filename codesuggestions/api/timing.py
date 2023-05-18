import functools
from time import time
from starlette_context import context


def timing(context_key):
    def decorator(f):
        @functools.wraps(f)
        def wrap(*args, **kwargs):
            start_time = time()

            try:
                result = f(*args, **kwargs)
            finally:
                end_time = time()
                context.data[context_key] = end_time - start_time

            return result
        return wrap
    return decorator
