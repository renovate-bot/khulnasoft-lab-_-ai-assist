import json
from time import sleep

import pytest
import responses
from jose import jwt

from codesuggestions.auth import GitLabAuthProvider, GitLabOidcProvider


@pytest.mark.parametrize(
    ("expiry_seconds", "wait", "want_calls"),
    [
        # second request is within expiry
        (3600, 1, 1),
        # second request is after expiry
        (1, 2, 2),
    ],
)
@responses.activate
def test_gitlab_auth_provider(expiry_seconds, wait, want_calls):
    ai_assist_response = responses.get(
        "http://test.com/ml/ai-assist",
        body='{"user_is_allowed": true, "third_party_ai_features_enabled": true}',
        status=200,
    )

    auth_provider = GitLabAuthProvider(
        base_url="http://test.com", expiry_seconds=expiry_seconds
    )

    user = auth_provider.authenticate("token")

    assert user is not None
    assert user.authenticated is True
    assert user.claims.is_third_party_ai_default is True

    sleep(wait)

    user = auth_provider.authenticate("token")

    assert user is not None
    assert user.authenticated is True
    assert user.claims.is_third_party_ai_default is True

    assert ai_assist_response.call_count == want_calls


class TestGitLabOidcProvider:
    # JSON Web Key can be generated via https://mkjwk.org/
    # Private key: X.509 PEM format
    # Public key: JWK format
    private_key = """
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

    public_key = {
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

    @responses.activate
    def test_gitlab_oidc_provider(self):
        well_known_response = responses.get(
            "http://test.com/.well-known/openid-configuration",
            body='{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
            status=200,
        )

        jwks_response = responses.get(
            "http://test.com/oauth/discovery/keys",
            body=json.dumps(self.public_key),
            status=200,
        )

        auth_provider = GitLabOidcProvider(base_url="http://test.com")

        token = jwt.encode(
            {
                "third_party_ai_features_enabled": True,
                "gitlab_realm": "self-managed",
            },
            self.private_key,
            algorithm="RS256",
        )
        user = auth_provider.authenticate(token)

        assert user is not None
        assert user.authenticated is True
        assert user.claims.is_third_party_ai_default is True
        assert user.claims.gitlab_realm == "self-managed"

        assert well_known_response.call_count == 1
        assert jwks_response.call_count == 1

    @responses.activate
    def test_gitlab_oidc_provider_missing_user_claims(self):
        well_known_response = responses.get(
            "http://test.com/.well-known/openid-configuration",
            body='{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
            status=200,
        )

        jwks_response = responses.get(
            "http://test.com/oauth/discovery/keys",
            body=json.dumps(self.public_key),
            status=200,
        )

        auth_provider = GitLabOidcProvider(base_url="http://test.com")

        token = jwt.encode(
            {"is_life_beautiful": True},
            self.private_key,
            algorithm="RS256",
        )
        user = auth_provider.authenticate(token)

        assert user is not None
        assert user.authenticated is True
        assert user.claims.is_third_party_ai_default is False
        assert user.claims.gitlab_realm == "saas"

        assert well_known_response.call_count == 1
        assert jwks_response.call_count == 1
