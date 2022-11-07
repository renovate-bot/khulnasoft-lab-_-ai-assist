import time

import pytest

from api.utils.cache import Cache

cache = Cache(expiry_seconds=2)


@pytest.mark.parametrize(
    "key",
    [
        "abcde",
        "12345",
        "a1b2c3d4e5"
    ]
)
def test_local_cache(key: str):
    set_key = cache.set_cached_token(key=key, value=True)
    retrieved = cache.get_cached_token(key=key)
    assert set_key == retrieved


@pytest.mark.parametrize(
    "key",
    ["abcde"]
)
def test_local_cache_expiration(key):
    set_key = cache.set_cached_token(key=key, value=True)
    assert set_key is True
    time.sleep(2)
    assert cache.get_cached_token(key=key) is False
