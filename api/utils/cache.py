import datetime
import os
import hashlib

import redis
from config.config import Config

config = Config()


class Cache:
    def __init__(self):
        if config.use_local_cache is True:
            self.cache = self.LocalCache()
        else:
            self.cache = redis.Redis(host='redis', port=6379, db=0, password=os.environ.get("REDIS_PASS"))
        self.expiry_seconds: int = 3600  # Oauth expires after 2 hours
        self.salt = os.urandom(16)

    class LocalCache:
        """
        Mocks Redis methods on a native Python in-memory dictionary object to reduce having to implement Redis.
        """
        def __init__(self):
            self.in_memory_cache = {}

        def set(self, name: str, value: str, ex: int) -> None:
            if self.in_memory_cache.get(name, None) is None:
                expiration = datetime.datetime.now() + datetime.timedelta(seconds=ex)
                self.in_memory_cache[name] = {"value": value, "ex": expiration}

        def exists(self, name: str) -> int:
            key = self.in_memory_cache.get(name)
            if key is not None:
                if key["ex"] > datetime.datetime.now():
                    # Key is still in cache and not expired
                    return 1
                else:
                    # Key is in cache but it's expired
                    del self.in_memory_cache[name]
                    return 0
            # Key does not exist in cache
            return False

    def set_cached_token(self, key: str) -> bool:
        """
        Insert a token into Redis
        :param key: Token, will be used for both key and value
        :return: bool
        """
        key = self._encode(key=key)
        self.cache.set(name=key, value=key, ex=self.expiry_seconds)
        return True

    def get_cached_token(self, key: str) -> bool:
        """
        Checks if a key exists in Redis and returns a boolean
        :param key:
        :return: bool
        """
        if self.cache.exists(self._encode(key)) > 0:
            return True
        return False

    def _encode(self, key: str) -> str:
        # return key
        hashed = str(hashlib.pbkdf2_hmac('sha256', key.encode(), self.salt, 10000))
        return hashed
