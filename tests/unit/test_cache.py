import pytest

from api.utils.cache import Cache

cache = Cache()


@pytest.mark.parametrize(
    "key",
    [
        "abcde",
        "12345",
        "a1b2c3d4e5"
    ]
)
def test_cache(key: str):
    set_key = cache.set_cached_token(key=key)
    retrieved = cache.get_cached_token(key=key)
    assert set_key == retrieved
