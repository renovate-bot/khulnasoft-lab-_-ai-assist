import json
import logging
import os
from time import sleep
from unittest.mock import patch

import pytest
import requests
import responses
from jose import jwt
from jose.exceptions import JWKError

from ai_gateway.auth import CompositeProvider, GitLabOidcProvider, LocalAuthProvider


class TestCompositeProvider:
    # JSON Web Key can be generated via https://mkjwk.org/
    # Private key: X.509 PEM format
    # Public key: JWK format
    private_key_ai_gateway_signing_key_test = """
-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC8fPBt+n0IIAAo
hqChw64BGRAsYtZo0J/B12yYp/2ZQHDYnOiYLnx9WJUwKg0/lyvNnuUAJ6Zjy2aF
x5Qa3dRLvT0G0jXsMdYyT9Dk5E1wi+0+whgKDxPNHuF3/BGLbXwu0ZcgEqUbMvfQ
EK70kWQinFq2nR52sj/+8HwlXhPCAX1r66f10Kxrc5SZ3DBcr7RYGxVycP9PBy9Z
psTh9RvmgOvCH5yEJoU4VsglilCL/HxFOQcN8daAY1m3kK5Bct2saT1wK95Scfdy
cAp6ZBz65PIvtmLMuQw3ndxzfJN1/oG9ge1TDL/OS+aW1VJChfTAWF8ur0OiBgs3
w24oU8AJAgMBAAECggEAARCFsMVSibuuiP3WU7Dk7Y1FROtZ39IP0dAWBTvrp/DX
Jd/SGpaYZuYNbd/K7TInrl9d1pGGULCVl57joqpukTYptIuEqrlO/9xa7jQDjkky
scvZqy9RyA9undc5lgfsPafJGqLoUk/ry/3JBPPeQw05g5mvg/GaKzRMOWvqLwqB
WFB6Hcoi+B8zD1SUtfJzp9zDQZ58m46vViGNxrNhkntHI8tOo5H2EQxdzUssSdTb
SZoLk90yzszYvGu8URPuPJbMdk8bWlrUpn1fASe166UVvQ7Y6AD/HMwA0zRhjKhD
h+CPBU1kE/bZAXUvlFdeQNG1rQTP/Tj3skWGAIhMhQKBgQDvWX+Sfb8XE0o9Bu6T
rsAZ/Mrlm4yjQaEV7LEJbMRSVm7qf8l2quLBdJ50TZM4lciKtzaoLynBgQ/4zczo
93nFrbqWnlVlxO7tMclgaSXM1mVXPiGB1w5TDVWjJsUiFFGZYK9xd17r6En3uco5
Zg6YyUHKCr06me4PWessVWuwRQKBgQDJmanST8KFFc1EiI1ohMllYhzxdoKbE10Q
9xL10hs3NGEJOH+J0RaFm2vQSMujDbbUXLFEl+P0fdhgodDwgrBP+/fVS791S0a/
y7Y/zotR88++Co4yz0WMmgkCRGiwqJnQuvncuaOJObotg45KFTJ1D5v0rWpaxhKf
Ra/0A9e29QKBgQCggtaIuQdjRC5vCq0IIRL22o5+uHfyK9sJRvfaqDRoO0qavCOx
DxyOO9Tfjf6C3f/k9sUSuL455IF/ixQ1z3C8XqtYwsnmO9E3BEJWA220FrtTbHkw
B7a1f6XEigV9uz6Vqz88yp6/ecHQ/aleIND9KUqTYexQ1lXNubF6w7Y6OQKBgQCc
J002HRez1CZR/l9h5PDGec+nbL9PdRkySd7Cz8LK6OR8qumHC5ChXriM9cXd/4Jt
TXr1gZ1NRKj0eIKJuQDug2H9MhYTuYIMj7MUC1041lxEfJKWYpwhgzKVMf3RUFcM
KbfeM2Crqy49kNgHJBIYQEXxqN1ngGLuQaE/pjZRfQKBgHWHBgh5kbruu1nyamdB
VZa/bwDiuM7xlEnBnhenLIvdJM1TqIczieNAZIXa7VCWVLSDaWvrtaq/h4i3v5kj
KXSvBjykZx92Su24sgukm3P7sT9hyEuerUFPUGt2axCrL8JNL4JXWC5/KkXJREjt
5OhnoClVD62lY8Bc90NTkMJe
-----END PRIVATE KEY-----
    """
    private_key_ai_gateway_validation_key_test = """
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC8YtGLWttDfdaR
cMRSsuHV0EljYIlS+B4gtCM8OplNKeTz9pxyIBlIaw83Z2FCAgC9tr1MlYZ5RoWq
hMiDOiC774hU5syrAlQhMHd8ijQo7fa8qTc08f1I+lkCbWRyFi3X15qf/80L60Ou
VhpsPzhs1mKYTCrY1c/MHnYdyf5VFNQvypFLvYoIrNemJueTLywZLS6+XsDphpIB
rBruTYkp+aP0Nn2t1ykATVySVvaeuuT85z80O72afyn7sPSpCtSTg28y/w3Nybjo
s1JQXw3zMTNX0D8JnvK9yVj+bJqh7giHluIBnrxwABGakw2Aj3qXt1rGM5BAlIyX
HLSsvdnJAgMBAAECggEAF3mWbE++9zTzBBZp5nbDSNQu0vcEgb3orjFYWzDfqdOM
w+BhHCEXGXS61Y/4uQOcIUfoZKbU1h+1OBuaC202x0iI4/CLREuS3XxnRVIJACkw
GBAbkKnPgtSsew6T59oO7iDANEwpWmD/ovW4jvDdWIWqDVmfdLwHDAPnOtzRCDus
Jx34OrqyTPoLJ7d/h2gJNN2r+qZll87MvOGhIPrBKgs1mJlxvcnIxHJarWrP/X43
4x9NBJ66xGWjbzJ+zKT/UhBQyQDNQoGGCBZwfPQcM1qelbUEOGoqBqWB1qLCFvfq
bdMoeZPoyM4MCNi18e8YYYMKmmfEjat6fVJLyb4B4QKBgQD6NUsFwTf+QdjU/vU3
9lN0LeA1BTpKb6s7UJS+/MHioTxv2ZWv/bmwrzrd8BtO6qYRDzcLTMbwkNSSz80+
rDs2pbVs4W92RmnHgvf73WiK0WhAQUqcQG8nX4BefETC9sKmiL5hkUja5U847A1i
V9DmNr7FLUVsk3I84mR3nQf06QKBgQDAvyyF73wzICUU4pGKqFFJPlLIkZcXbdmt
AhamTKLQwNSxnK6b6MgzQXYkqLbgJ2aau5v8wFrQVURLZx8+OLjAE2S6CrK0RLPO
mT9rlJNhSjx3Lacq9Lc8Pk5a5HEB4dSvEQk6riHPWo4GWpcmsIrw6X12C1XkR1xa
dNhNnIcx4QKBgHlZaYZj/K0i8G/1K6c1n6oEKe5tF6VMXYbKASpT2hD5VB+HLtMJ
QosPoYRMVGJE6b/yWibv2LiJ9Z8yi3+u9pT9b21cNLvvUJRDz9PmwTI6d85aHD6F
/aLh7Zdlu8+28Bbm0TbuyJ/pgS/BRIiCwL02pfVpjHcpV8lxn3pnvZkpAoGAJKX0
5D6F4f6xrkfqHnAkjIWiHeq4zMahRekIv3QA3SpdBqxg8toO/tfqi8vcgcBcHP2h
CizU15nu01t3MFB+qF7HnywbkHUjrxuqWF02rJ/94Tc3+s3u7TB3m4amChKTavoV
RCgJ27A/Iuwko0GcGXR7228KVM5QvA5NdmxVtGECgYEAorK1bE8j1u8sjXkpbyxh
uMQ15NX72v03rOSjEiglkVYVPM6+jA4A5TZYQLEYfyGDjw/twsxNwHLi6hzoqkJv
C7OH109IoruXRyRUXjizfZJCtVnE6sYCXgeJMQgIivwD4WD+5jkMunQBh8v3/N8j
x9RTl7Z7UubXvhAXD6+4uh0=
-----END PRIVATE KEY-----
    """
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

    forged_private_key = """
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCB19bRw7ut/Rt6
LGxtPhGJ4/t8QS31mem4djNYteDKLTAc6YykJos8Gv+KzTioO2LBwHXF7QZPLcJW
FrRewARjwWQrVnJm6OCFaGs4n9/RE+o2JCB2C/DBR+mLKoGCSPH+ts8i87zwviRw
LD3G4XkPU2cyCAyIIhqz/9rJ7MLCbtAZYWeWzhyX/BZt/nHVT/HGEkZx7HMoDeir
DjIlJANoNLMYSfdyWNyeRZocZpDjGTR/k39Ycvlvwif8KE6hGfPjFAytdLqw/tP2
wDpNoFPja7RmI6OB/9C2NReLtiCiY9U/dxqZdbxgmxDSRs8xG20OL87TmbNFiAkO
GhkK8znfAgMBAAECggEAFPCDA1cmiDueV8icP83XtD4hC3vTdp04tPfS9gZ21wQM
k00S0lIo+Ct+dJr9/Rt+pLJuC9pavyApDAsjUXhz/MZuahLJ5lC+DKW3TO1zgdtN
VSfkWEU6sWDwh/H16tXquOIwa3mVSdnQNIrd95nbFR6lMMtdggLF/atQVGorSoGr
ldDf3aA76zcTC4k6NxZwxUXZuYn76iconSSlEoqYoE15/yT/j1Or4SCXhAvixzox
n8DPE6w7wbp43fTbghCkd4mU19A06dNHXD1x92eOXEhGk8GuWGhm4yypOryZRb7y
94bytD3qceZ1c8Dr/3F54W9LFgCJC9+3OCwMjgOsiQKBgQC8TwdfR8pQIK62yOqQ
j7xuYm5TQpeGSXUSZ7j8WaUVhb3K7sDBlOcp/NZ7wVT+HxEK7rgFcg+4f+mwfQpG
hVvMZZKBkLtv2JZA0AoJqxqS33y8T1o6Qf0c+ig9O8HlUzjjr0XeGYEPasIQiE50
12Yyw/9WEX7dt+WFJzhAl6HjBQKBgQCwhI6/K8UNFC/XSSALmpQrsGiO5bRfUGnU
AMoTM6Iekqa2o0yKwSXB618qQeR6AiMf83pZ18ZDC4y8LdwjHhayOm9LkaQhD98o
/7HyOA4ao85Yg+Aavr66huTzIUhm8tQ2XaS+kPHGzzdcU5nxRHpozDtb4PLFK19x
X/wPYkjGkwKBgGyTtNB/eGvTLGpAVt+bwS50muBvGSdY26QNImB+3+0U/GYyW/pC
fTd8jb81rmgISa9gDcM2DVJ4jqowrugSpOep+VuztB+9ZoVgbyk7+0qMikOaDZBh
1CwNIX6NIjO0VK0TttllI0FccSFPNs2wFUFYObXKyLfW/QRVpN25kKJxAoGBAJ1G
rs2E2UELAIlonUXZiDXZK4BPCMR4KKL9xQ8QzV1CO4q1u2hSKis7ZYKITWOsbdF1
JknQqNVqAA5XKjKcB4rr5+hELyJKOwMTuGBiM6bm1t8lOVN7KwOVV3+N4y3fJNf/
3d7x1IrYbLI1xw8ifZLMjgMSTh0BxTuGU1b9smxDAoGAMLwdvGy8OuE6OjaKSwFh
DuAGV7NRJlhf1Ga2b1dORFMdhuEUlGkxuH5+Tw+uLylgBE91gJMR+fgSQ+d3aKOq
2rSHdIXOeCpVoIrSW6xY7a1HnzNBI9o/afaGcYTpuEhlgCdGxDr9sdqP495pUNsI
UGw3kIW+604fnnXLDm4TaLA=
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
        "kid": "ZoObadsnUfqW_C_EfXp9DM6LUdzl0R",
        "n": "stiq29kurlOhBUZuBT4Yhths3D0pc98nTAX58CGKRyJQuZ531vl2JuMaJk7zValW-JIzZ4bCAe"
        "J84_DaBUWg_L9a0_0hdMGT5R-U0Mo_YQGZ-9y5ECcN5rstKt_Cy64RLdFajSI8MR-uTISYAendkXhoE"
        "vajw9D52S5RVFXNbR9U8S13x6dJ6Jvhe6Oj3fHbYoaloG54p8Y7j3b2Y9tAr3hQGqjYIK3a6srI_fff5"
        "LBjOhbqlVEbRcyiRxsso4NGVv_MhsmFkF61uyfDwclTDjT1K1xDzWdnURI5cQVzMXjLLWTw9AMoAKosZ"
        "lyP1hDFgt6OWvC-liWdrejZjCTz3w",
    }

    claims = {
        "gitlab_realm": "self-managed",
        "scopes": ["code_suggestions"],
        "issuer": "https://customers.gitlab.com",
    }

    ai_gateway_audience = {"aud": "gitlab-ai-gateway"}

    @responses.activate
    @pytest.mark.parametrize(
        "config_response_body,jwks_url,jwks_response_body,problematic_provider,expected_jwks_call_count",
        [
            ('{"jwks_uri": ""}', "", '{"keys": []}', [], 0),
            ("{}", "", "{}", [], 0),
            (
                requests.exceptions.RequestException("OIDC config request failed"),
                "",
                "{}",
                "CustomersDot",
                0,
            ),
            (
                '{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
                "http://test.com/oauth/discovery/keys",
                '{"keys": []}',
                None,
                1,
            ),
            (
                '{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
                "http://test.com/oauth/discovery/keys",
                "{}",
                None,
                1,
            ),
            (
                '{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
                "http://test.com/oauth/discovery/keys",
                requests.exceptions.RequestException("JWKS request failed"),
                "CustomersDot",
                1,
            ),
        ],
    )
    def test_broken_gitlab_oidc_provider_responses(
        self,
        config_response_body,
        jwks_url,
        jwks_response_body,
        problematic_provider,
        expected_jwks_call_count,
    ):
        # We only need one endpoint to simulate failures; this one will be successful.
        well_known_test_response = responses.get(
            "http://test.com/.well-known/openid-configuration",
            body='{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
            status=200,
        )
        jwks_test_response = responses.get(
            "http://test.com/oauth/discovery/keys",
            body=f'{{"keys": [{json.dumps(self.public_key_test)}]}}',
            status=200,
        )

        # This endpoint will contain various faulty responses.
        well_known_customers_response = responses.get(
            "http://customers.test.com/.well-known/openid-configuration",
            body=config_response_body,
            status=200,
        )
        jwks_customers_response = responses.get(
            jwks_url,
            body=jwks_response_body,
            status=200,
        )

        auth_provider = CompositeProvider(
            [
                GitLabOidcProvider(
                    oidc_providers={
                        "Gitlab": "http://test.com",
                        "CustomersDot": "http://customers.test.com",
                    }
                )
            ]
        )

        token = jwt.encode(
            {},
            self.private_key_test,
            algorithm=CompositeProvider.RS256_ALGORITHM,
        )

        with patch("ai_gateway.auth.providers.log_exception") as mock_log_exception:
            user = auth_provider.authenticate(token)

            if problematic_provider and isinstance(config_response_body, Exception):
                mock_log_exception.assert_called_once_with(
                    config_response_body, {"oidc_provider": problematic_provider}
                )
            elif problematic_provider and isinstance(jwks_response_body, Exception):
                mock_log_exception.assert_called_once_with(
                    jwks_response_body, {"oidc_provider": problematic_provider}
                )

        assert user is not None

        # The successful provider calls.
        assert well_known_test_response.call_count == 1
        assert jwks_test_response.call_count == 1

        # The unsuccessful provider calls.
        assert well_known_customers_response.call_count == 1
        assert jwks_customers_response.call_count == expected_jwks_call_count

    @responses.activate
    @pytest.mark.parametrize(
        "jwks_response_body",
        [
            "{}",
            '{"keys": []}',
        ],
    )
    def test_no_jwks_available_raises_error(
        self,
        jwks_response_body,
    ):
        # pylint: disable=direct-environment-variable-reference
        os.environ.pop("AIGW_SELF_SIGNED_JWT__SIGNING_KEY", None)
        os.environ.pop("AIGW_SELF_SIGNED_JWT__VALIDATION_KEY", None)
        # pylint: enable=direct-environment-variable-reference

        well_known_test_response = responses.get(
            "http://test.com/.well-known/openid-configuration",
            body='{"jwks_uri": "http://test.com/oauth/discovery/keys"}',
            status=200,
        )
        jwks_test_response = responses.get(
            "http://test.com/oauth/discovery/keys",
            body=jwks_response_body,
            status=200,
        )

        auth_provider = CompositeProvider(
            [
                LocalAuthProvider(),
                GitLabOidcProvider(
                    oidc_providers={
                        "Gitlab": "http://test.com",
                    }
                ),
            ]
        )
        token = jwt.encode(
            {},
            self.private_key_test,
            algorithm=CompositeProvider.RS256_ALGORITHM,
        )

        user = None
        with pytest.raises((CompositeProvider.CriticalAuthError, JWKError)):
            user = auth_provider.authenticate(token)

        assert user is None

        assert well_known_test_response.call_count == 1
        assert jwks_test_response.call_count == 1

    @responses.activate
    @pytest.mark.parametrize(
        "jwt_signing_key,jwt_validation_key,key_to_sign,kid",
        [
            (
                None,
                private_key_ai_gateway_validation_key_test,
                private_key_ai_gateway_validation_key_test,
                "gitlab_ai_gateway_validation_key",
            ),
            (
                private_key_ai_gateway_signing_key_test,
                None,
                private_key_ai_gateway_signing_key_test,
                "gitlab_ai_gateway_signing_key",
            ),
        ],
    )
    def test_missing_one_environment_variable(
        self,
        jwt_signing_key,
        jwt_validation_key,
        key_to_sign,
        kid,
    ):
        # pylint: disable=direct-environment-variable-reference
        if jwt_signing_key:
            os.environ["AIGW_SELF_SIGNED_JWT__SIGNING_KEY"] = jwt_signing_key
        else:
            os.environ.pop("AIGW_SELF_SIGNED_JWT__SIGNING_KEY", None)
        if jwt_validation_key:
            os.environ["AIGW_SELF_SIGNED_JWT__VALIDATION_KEY"] = jwt_validation_key
        else:
            os.environ.pop("AIGW_SELF_SIGNED_JWT__VALIDATION_KEY", None)
        # pylint: enable=direct-environment-variable-reference

        auth_provider = CompositeProvider([LocalAuthProvider()])
        token = jwt.encode(
            self.claims | self.ai_gateway_audience,
            key_to_sign,
            algorithm=CompositeProvider.RS256_ALGORITHM,
        )

        user = auth_provider.authenticate(token)

        assert user is not None
        assert user.authenticated

        cached_keys = auth_provider.cache.get(auth_provider.CACHE_KEY).value
        assert len(cached_keys["keys"]) == 1
        assert cached_keys["keys"][0]["kid"] == kid

    def test_missing_environment_variables_error(self):
        # pylint: disable=direct-environment-variable-reference
        os.environ.pop("AIGW_SELF_SIGNED_JWT__SIGNING_KEY", None)
        os.environ.pop("AIGW_SELF_SIGNED_JWT__VALIDATION_KEY", None)
        # pylint: enable=direct-environment-variable-reference

        auth_provider = CompositeProvider([LocalAuthProvider()])
        token = jwt.encode(
            self.claims | self.ai_gateway_audience,
            self.private_key_ai_gateway_validation_key_test,
            algorithm=CompositeProvider.RS256_ALGORITHM,
        )

        user = None
        with pytest.raises(CompositeProvider.CriticalAuthError):
            user = auth_provider.authenticate(token)

        assert user is None

    @responses.activate
    @pytest.mark.parametrize(
        "private_key_used_to_sign,claims,gitlab_realm,authenticated",
        [
            (
                private_key_test,
                claims | ai_gateway_audience,
                "self-managed",
                True,
            ),
            (
                private_key_ai_gateway_signing_key_test,
                claims | ai_gateway_audience,
                "self-managed",
                True,
            ),
            (
                private_key_ai_gateway_validation_key_test,
                claims | ai_gateway_audience,
                "self-managed",
                True,
            ),
            (
                private_key_customers,
                claims | ai_gateway_audience,
                "self-managed",
                True,
            ),
            (
                forged_private_key,
                claims | ai_gateway_audience,
                "",
                False,
            ),
            (
                private_key_test,
                {
                    "is_life_beautiful": True,
                    "scopes": ["code_suggestions"],
                    "gitlab_realm": "saas",
                }
                | ai_gateway_audience,
                "saas",
                True,
            ),
        ],
    )
    def test_composite_provider(
        self, private_key_used_to_sign, claims, gitlab_realm, authenticated
    ):
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

        # pylint: disable=direct-environment-variable-reference
        os.environ["AIGW_SELF_SIGNED_JWT__SIGNING_KEY"] = (
            self.private_key_ai_gateway_signing_key_test
        )
        os.environ["AIGW_SELF_SIGNED_JWT__VALIDATION_KEY"] = (
            self.private_key_ai_gateway_validation_key_test
        )
        # pylint: enable=direct-environment-variable-reference

        auth_provider = CompositeProvider(
            [
                LocalAuthProvider(),
                GitLabOidcProvider(
                    oidc_providers={
                        "Gitlab": "http://test.com",
                        "CustomersDot": "http://customers.test.com",
                    }
                ),
            ]
        )

        token = jwt.encode(
            claims,
            private_key_used_to_sign,
            algorithm=CompositeProvider.RS256_ALGORITHM,
        )
        user = auth_provider.authenticate(token)

        assert user is not None
        assert user.authenticated is authenticated
        assert user.claims.gitlab_realm == gitlab_realm

        if authenticated:
            assert user.claims.scopes == ["code_suggestions"]

        assert well_known_test_response.call_count == 1
        assert well_known_customers_response.call_count == 1
        assert jwks_test_response.call_count == 1
        assert jwks_customers_response.call_count == 1

        cached_keys = auth_provider.cache.get(auth_provider.CACHE_KEY).value

        assert len(cached_keys["keys"]) == 4
        assert [key["kid"] for key in cached_keys["keys"]] == [
            "gitlab_ai_gateway_signing_key",
            "gitlab_ai_gateway_validation_key",
            "MFRZ2Sp4sCciuzxArGCtNP5w2X716R6prptJqYHpFBw",
            "ZoObadsnUfqW_C_EfXp9DM6LUdzl0R",
        ]
