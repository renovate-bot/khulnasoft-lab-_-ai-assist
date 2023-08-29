import json
from time import sleep

import pytest
import responses
from jose import jwt

from codesuggestions.auth import GitLabOidcProvider


class TestGitLabOidcProvider:
    # JSON Web Key can be generated via https://mkjwk.org/
    # Private key: X.509 PEM format
    # Public key: JWK format
    private_key_test = """
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCAzC66XB4Fppc0
QFmrujl0AnGA7ffgnXvVJTdswIlDrYH/2R5F05ekS0Fzn/tFksyx4RZuakQu4SiP
K0FXipmpkF48wGbkSHLQObTdYmL0A6kF9VWVd29N2hun+gJNnvL32TPG6CEepIvH
UIgkfvvGQCSnFJW8QR+3+U29JsNYRUPsZRKvgV/z1zQt6ZLeBUg1sm5xA7qLsmjH
mThfJCosnmdKoYvZbapmQGhpXcjtmlCSVeibVcosXsnrAiXn7Cs+9caWNvKZMnlT
lY+kZwliOqI/BusFVUIr2/sWi/MsN0UIrtWsqlTlL/JVwqsg6bwANqOFMByZxIRk
GgTCyFO3AgMBAAECggEAAtJ2g6bZEY6g6Ygvbs/ZymzzR7vvHoDU4cq6+CsP1ufK
XWzIeQc132e2u23Z96BL0+n2r9ysOcq9NMXh3KUw0MJVDke4+W+M9HsPN3qcaHRc
E8FYarn/Oll5Gakku8ar1DpyI/2aHC3G0ks1cHdH1QQ6yV5uGX3j0AgqZ+adiSWT
LfbXepCedQp+w29AgnaKBfFb2tVl/5DgCBsR2GDeveFwgmDEZiuS+Mz44XYpLkaF
FWXKfwA+HjOSI+8qUgSdGj/AGvK3E4YZZHqJ8eXLXXHPxL+UQFEgfST1zJQFF51B
X1CfrG5lAgrxra32vsuj36T5uFaXUhdUcSJddQJsoQKBgQDpUGixsX1Lynn6JzwQ
1MT/CvVxMF57Af2xFeggfwpiTCa8eKw/vjxXcFoQIbnTSr+ugGuyPEHwmImLR0Qj
FQE4kbM7vwuRKtJVCkqppeBA4xAmA3qwW5/YLoVO1Qp4dqPWSaczQJeNM8+4+Z5C
tD88NPOjzkTTlpNxw84nnI1FrQKBgQCNUi1TMVdUCz6ZPINyCtjNAMam7yEtiegh
xRr/A/v83ku+Gy3ZfOQbRcarRxkmDHHsPwpSCnHHUIBTaFKlMqmAEl3u58fmYVSR
pJSgvxuYJGS+GAJsmQ/VQvaYA9sH93mV3VZBQOtNrRsalfdoSgMo7KArsSisArgb
0ElDO6UDcwKBgFmVsl1oVT/gwu02W23rBKkZQBzyAZUhspNoYfT4UrhjnQwJGbpw
BSNd1HcVPBDRRsBuNuv9DySerVF5T8RYsFtUNoneVUasNo7IoNp7Apxnky/FbjqB
M+MCGdWnH5oZk9cX+MdJKefh2QShdA8QvqcTfemLrgnAa2TnViUHi4cRAoGAcgyx
24PkcEUq3cwCYNT0Jm3L5Aj0g6XaGvbRVKFIich05BVXKUArbv8e2Ddmylgc0IYH
tDINpMcI6Uc1+3Apbtxjxlxz7S77axahhCD3Cg/E5czGmBHmvzttez0RVRqZmyKn
a74Sp/td9lS0+AtTBYIBuYEdy8PeBURQ+9t0zpUCgYEAhefNoiILN3OO+fgv/L+7
meZ76Z9wPG3CuNPtQqgG4zHy57z9bZMcDMHnL60NV6liH0E8CxGQ07NGjI3qdfs9
/Osy0rv4v39yWpNt4M6d7lqyZoIydfrhU0RQqCtRUoZuql73R0KpD89r9uiaLaxd
cmOY8Rcamyo/giOO9+jYMaU=
-----END PRIVATE KEY-----
    """
    private_key_customers = """
-----BEGIN PRIVATE KEY-----
MIIEwAIBADANBgkqhkiG9w0BAQEFAASCBKowggSmAgEAAoIBAQCy2Krb2S6uU6EF
Rm4FPhiG2GzcPSlz3ydMBfnwIYpHIlC5nnfW+XYm4xomTvNVqVb4kjNnhsIB4nzj
8NoFRaD8v1rT/SF0wZPlH5TQyj9hAZn73LkQJw3muy0q38LLrhEt0VqNIjwxH65M
hJgB6d2ReGgS9qPD0PnZLlFUVc1tH1TxLXfHp0nom+F7o6Pd8dtihqWgbninxjuP
dvZj20CveFAaqNggrdrqysj999/ksGM6FuqVURtFzKJHGyyjg0ZW/8yGyYWQXrW7
J8PByVMONPUrXEPNZ2dREjlxBXMxeMstZPD0AygAqixmXI/WEMWC3o5a8L6WJZ2t
6NmMJPPfAgMBAAECggEBAIHRejw46oSJmcDtfaD6kO0YnfRDxRohqjGpyOHARtIZ
m4UQ/SYjT9ssT+fsuP69+65U2VFVZO/fSg5e3rKi9xdfgvuLq0RH2yWehfQESnsM
oYxLjF2oK5QG2+NaJtiX0kpyw4rchdqWh3ttZ7VD35vfTZQuSXMy9pjp4QkZexKw
SuZ0fATQ7Iq0GXhFgKS7a8fK5ErLWRu5rZ6T0tqykoXFhvb3e1Wkr7KQZ0rfAI37
J4dzA6bs7sBufD4nenJ91VKoQuVytT4XWeVmHrkN1uRVbsKyXN0zwMBd0rbyoRhK
YpTvgJ+9vjrmuerheJ/UQZADCNwsk8lvqiBSoSPJRwECgYEA56+YxehNjJB+vzox
X9nqZ5dN6/yb2vD5E5+Kdy1QS8i3KDc5NDKIKwOj369AJ/6yO3HIT84nAA+hyaQV
bZR8TAn1HcjyVneS8xUv3yHUf6OtX0fHzBoQ0SBHV8de7kb141J1A61eZXigWdQf
xyPaU1Gd/FVEs0e/Ms5L3fGyzV8CgYEAxZ1/vKp+lDdHO1eFaGt7OyRyjFR5uINg
qh41BSdvUQISBMLcIjy0iQwz3X/IsGhVJLpZnq4KcvbwjmxnBfALe6V3W4p0lkrQ
KZSMbj5CQ9EvHUIWhH/63EcXDiF0lE1P02oyIE6GgXHmPzPxIm4L9+R/woS7CtwJ
cIIGrU/g6YECgYEApJS1pujlpdPDZd5V0qw/eoUeAmR45qfFmC3+M7jJ0HvtuDC6
fxziZAP69lxM69xfCiFC1YYxaDayjEX2Pth7D50HNbhYhhn9FpbXYd9rT5ya/RNF
l/RwO859P5rOEd/wriIWI52Vb+mnpwgr5s/OON/CpcyAuAZgiRvJAwm+JWsCgYEA
nFgxWYjiQE1Ds/VYfPacNnxtjAzBiHOYpL7lX8CFV2f17YJlO0kf5FWdKx8QHlFN
G5O7l8lGRxKL24J0N8RksVyGBAyUlNt3uY3nVMk5EAKN7e39drLyPBiaavmZDEPm
ZfNc2SaHUB5W9aYYnw7FtUg4tCLjAIJ5jWOx+Kh73wECgYEAqYJrXZLG3X0gY3hw
2kS5BpWEyoQJDVgA8lxK+ND22MwtN0NMDbmp4h3HifkXrZ2EKcTwgUx4revqtv3I
Piz6UxMdEyvOKmcujB6dYNem3LONaaaxDEYcPwApNSH1ZCKHnM8zBBI6EolIkNhZ
0GGnJpqAOqw0uWLx1Wp+fQYu0H0=
-----END PRIVATE KEY-----
    """

    public_key_test = {
        "kty": "RSA",
        "e": "AQAB",
        "use": "sig",
        "kid": "MFRZ2Sp4sCciuzxArGCtNP5w2X716R6prptJqYHpFBw",
        "alg": "RS256",
        "n": "gMwuulweBaaXNEBZq7o5dAJxgO334J171SU3bMCJQ62B_9keRdOXpEtBc5_7RZLMseEWbm"
        "pELuEojytBV4qZqZBePMBm5Ehy0Dm03WJi9AOpBfVVlXdvTdobp_oCTZ7y99kzxughHqSL"
        "x1CIJH77xkAkpxSVvEEft_lNvSbDWEVD7GUSr4Ff89c0LemS3gVINbJucQO6i7Jox5k4Xy"
        "QqLJ5nSqGL2W2qZkBoaV3I7ZpQklXom1XKLF7J6wIl5-wrPvXGljbymTJ5U5WPpGcJYjqiP"
        "wbrBVVCK9v7FovzLDdFCK7VrKpU5S_yVcKrIOm8ADajhTAcmcSEZBoEwshTtw",
    }

    public_key_customers = {
        "kty": "RSA",
        "e": "AQAB",
        "use": "sig",
        "alg": "RS256",
        "n": "stiq29kurlOhBUZuBT4Yhths3D0pc98nTAX58CGKRyJQuZ531vl2JuMaJk7zValW-JIzZ4bCAe"
        "J84_DaBUWg_L9a0_0hdMGT5R-U0Mo_YQGZ-9y5ECcN5rstKt_Cy64RLdFajSI8MR-uTISYAendkXhoE"
        "vajw9D52S5RVFXNbR9U8S13x6dJ6Jvhe6Oj3fHbYoaloG54p8Y7j3b2Y9tAr3hQGqjYIK3a6srI_fff5"
        "LBjOhbqlVEbRcyiRxsso4NGVv_MhsmFkF61uyfDwclTDjT1K1xDzWdnURI5cQVzMXjLLWTw9AMoAKosZ"
        "lyP1hDFgt6OWvC-liWdrejZjCTz3w",
    }

    claims = {
        "third_party_ai_features_enabled": True,
        "gitlab_realm": "self-managed",
    }

    @responses.activate
    @pytest.mark.parametrize(
        "private_key,claims,third_party,gitlab_realm",
        [
            (private_key_test, claims, True, "self-managed"),
            (private_key_customers, claims, True, "self-managed"),
            (private_key_test, {"is_life_beautiful": True}, False, "saas"),
        ],
    )
    def test_gitlab_oidc_provider(self, private_key, claims, third_party, gitlab_realm):
        well_known_test_response = responses.get(
            "http://test.com/.well-known/openid-configuration",
            body='{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
            status=200,
        )

        well_known_customers_response = responses.get(
            "http://customers.test.com/.well-known/openid-configuration",
            body='{"jwks_uri": "http://customers.test.com/oauth/discovery/keys"}',
            status=200,
        )

        jwks_test_response = responses.get(
            "http://test.com/oauth/discovery/keys",
            body=f'{{"keys": [{json.dumps(self.public_key_test)}]}}',
            status=200,
        )

        jwks_customers_response = responses.get(
            "http://customers.test.com/oauth/discovery/keys",
            body=f'{{"keys": [{json.dumps(self.public_key_customers)}]}}',
            status=200,
        )

        auth_provider = GitLabOidcProvider(
            oidc_providers={
                "Gitlab": "http://test.com",
                "CustomersDot": "http://customers.test.com",
            }
        )

        token = jwt.encode(
            claims,
            private_key,
            algorithm="RS256",
        )
        user = auth_provider.authenticate(token)

        assert user is not None
        assert user.authenticated is True
        assert user.claims.is_third_party_ai_default is third_party
        assert user.claims.gitlab_realm == gitlab_realm

        assert well_known_test_response.call_count == 1
        assert well_known_customers_response.call_count == 1
        assert jwks_test_response.call_count == 1
        assert jwks_customers_response.call_count == 1

        cached_keys = auth_provider.cache.get(auth_provider.CACHE_KEY).value

        assert len(cached_keys["keys"]) == 2
