import urllib.parse
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from jose import JWTError, jwk, jwt
from jose.exceptions import JWKError

from ai_gateway.cloud_connector.cache import LocalAuthCache
from ai_gateway.cloud_connector.config import CloudConnectorConfig
from ai_gateway.cloud_connector.user import CloudConnectorUser, UserClaims
from ai_gateway.tracking.errors import log_exception

__all__ = [
    "AuthProvider",
    "GitLabOidcProvider",
    "CompositeProvider",
    "LocalAuthProvider",
    "JwksProvider",
]

REQUEST_TIMEOUT_SECONDS = 10


class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, *args, **kwargs) -> CloudConnectorUser:
        pass


class JwksProvider:
    def __init__(self, log_provider) -> None:
        self.logger = log_provider.getLogger("cloud_connector")

    @property
    def cloud_connector_service_name(self) -> str:
        return CloudConnectorConfig().service_name

    def jwks(self, *args, **kwargs) -> dict:
        cached_jwks = self.cached_jwks(*args, **kwargs)
        if cached_jwks and len(cached_jwks) > 0:
            return cached_jwks

        new_jwks = self.load_jwks(*args, **kwargs)
        self._log_keyset_update(new_jwks)
        return new_jwks

    @abstractmethod
    def cached_jwks(self, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def load_jwks(self, *args, **kwargs) -> dict:
        pass

    def _log_keyset_update(self, jwks):
        kids = [key["kid"] for key in jwks["keys"]]
        self.logger.info("JWKS refreshed", kids=kids, provider=self.__class__.__name__)


class CompositeProvider(JwksProvider, AuthProvider):
    RS256_ALGORITHM = "RS256"
    SUPPORTED_ALGORITHMS = [RS256_ALGORITHM]
    CACHE_KEY = "jwks"

    class CriticalAuthError(Exception):
        pass

    def __init__(
        self, providers: list[JwksProvider], log_provider, expiry_seconds: int = 86400
    ):
        super().__init__(log_provider)
        self.providers = providers
        self.expiry_seconds = expiry_seconds
        self.cache = LocalAuthCache()

    def authenticate(self, token: str) -> CloudConnectorUser:
        jwks = self.jwks()
        is_allowed = False
        gitlab_realm = ""
        subject = ""
        issuer = ""
        scopes = []
        duo_seat_count = ""
        gitlab_instance_id = ""

        if len(jwks.get("keys", [])) == 0:
            raise self.CriticalAuthError(
                "No keys founds in JWKS; are OIDC providers up?"
            )

        try:
            jwt_claims = jwt.decode(
                token,
                jwks,
                audience=self.cloud_connector_service_name,
                algorithms=self.SUPPORTED_ALGORITHMS,
            )
            gitlab_realm = jwt_claims.get("gitlab_realm", "")
            subject = jwt_claims.get("sub", "")
            issuer = jwt_claims.get("iss", "")
            scopes = jwt_claims.get("scopes", [])
            duo_seat_count = str(jwt_claims.get("duo_seat_count", ""))
            gitlab_instance_id = jwt_claims.get("gitlab_instance_id", "")

            is_allowed = True
        except JWTError as err:
            log_exception(err)

        return CloudConnectorUser(
            authenticated=is_allowed,
            claims=UserClaims(
                gitlab_realm=gitlab_realm,
                scopes=scopes,
                subject=subject,
                issuer=issuer,
                duo_seat_count=duo_seat_count,
                gitlab_instance_id=gitlab_instance_id,
            ),
        )

    def cached_jwks(self) -> dict:
        jwks_record = self.cache.get(self.CACHE_KEY)
        if jwks_record and jwks_record.exp > datetime.now():
            return jwks_record.value

        return defaultdict()

    def load_jwks(self) -> dict:
        jwks: dict[str, list] = defaultdict()
        jwks["keys"] = []
        for provider in self.providers:
            try:
                provider_jwks = provider.jwks()
                jwks["keys"] += provider_jwks["keys"]
            except Exception as e:
                log_exception(e)

        self._cache_jwks(jwks)
        return jwks

    def _cache_jwks(self, jwks):
        exp = datetime.now() + timedelta(seconds=self.expiry_seconds)
        self.cache.set(self.CACHE_KEY, jwks, exp)


class LocalAuthProvider(JwksProvider):
    ALGORITHM = CompositeProvider.RS256_ALGORITHM

    def __init__(self, log_provider, signing_key: str, validation_key: str) -> None:
        super().__init__(log_provider)
        self.signing_key = signing_key
        self.validation_key = validation_key

    def cached_jwks(self) -> dict:
        return defaultdict()

    def load_jwks(self) -> dict:
        jwks: dict[str, list] = defaultdict(list)
        jwks["keys"] = []

        try:
            signing_key = (
                jwk.RSAKey(
                    algorithm=self.ALGORITHM,
                    key=self.signing_key,
                )
                .public_key()
                .to_dict()
            )
            signing_key.update(
                {
                    "kid": f"{self.cloud_connector_service_name}_signing_key",
                    "use": "sig",
                }
            )
        except JWKError as e:
            log_exception(e)

        try:
            validation_key = (
                jwk.RSAKey(
                    algorithm=self.ALGORITHM,
                    key=self.validation_key,
                )
                .public_key()
                .to_dict()
            )
            validation_key.update(
                {
                    "kid": f"{self.cloud_connector_service_name}_validation_key",
                    "use": "sig",
                }
            )
        except JWKError as e:
            log_exception(e)

        jwks["keys"].append(signing_key)
        jwks["keys"].append(validation_key)

        return jwks


class GitLabOidcProvider(JwksProvider):
    def __init__(self, log_provider, oidc_providers: dict[str, str]):
        super().__init__(log_provider)
        self.oidc_providers = oidc_providers

    def cached_jwks(self) -> dict:
        return defaultdict()

    def load_jwks(self) -> dict:
        jwks = defaultdict(list)

        for oidc_provider, base_url in self.oidc_providers.items():
            well_known = self._fetch_well_known(oidc_provider, base_url)
            for k, v in self._fetch_jwks(oidc_provider, well_known).items():
                jwks[k].extend(v)

        if jwks:
            return jwks

        return {}

    def _fetch_well_known(self, oidc_provider, base_url: str) -> dict:
        end_point = "/.well-known/openid-configuration"
        url = urllib.parse.urljoin(base_url, end_point)

        well_known = {}
        try:
            res = requests.get(url=url, timeout=REQUEST_TIMEOUT_SECONDS)
            well_known = res.json()
        except requests.exceptions.RequestException as err:
            log_exception(err, {"oidc_provider": oidc_provider})

        return well_known

    def _fetch_jwks(self, oidc_provider, well_known) -> dict:
        url = well_known.get("jwks_uri")
        if not url:
            return {}

        jwks = {}

        try:
            res = requests.get(url=url, timeout=REQUEST_TIMEOUT_SECONDS)
            jwks = res.json()
        except requests.exceptions.RequestException as err:
            log_exception(err, {"oidc_provider": oidc_provider})

        return jwks
