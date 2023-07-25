import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, NamedTuple, Optional

import regex

# noinspection PyProtectedMember
from detect_secrets.core.scan import PotentialSecret, _process_line_based_plugins
from detect_secrets.plugins.keyword import QUOTES_REQUIRED_DENYLIST_REGEX_TO_GROUP
from detect_secrets.settings import transient_settings

__all__ = [
    "Detected",
    "DetectorKind",
    "BaseDetector",
    "DetectorRegex",
    "DetectorRegexEmail",
    "DetectorRegexIPV6",
    "DetectorRegexIPV4",
    "DetectorBasicAuthSecrets",
    "DetectorTokenSecrets",
    "DetectorKeywordsSecrets",
]

email_pattern = r"""
    (?<= ^ | [\b\s@,?!;:)('".\p{Han}<] )
    (
      [^\b\s@?!;,:)('"<]+
      @
      [^\b\s@!?;,/]*
      [^\b\s@?!;,/:)('">.]
      \.
      \p{L} \w{1,}
    )
    (?= $ | [\b\s@,?!;:)('".\p{Han}>] )
"""

ipv4seg = r"(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
ipv4addr = r"(?:(?:" + ipv4seg + r"\.){3,3}" + ipv4seg + r")"
ipv6seg = r"(?:(?:[0-9a-fA-F]){1,4})"

# Examples of emails identified:
# 1:2:3:4:5:6:7:8
# 1:: 1:2:3:4:5:6:7::
# 1::8 1:2:3:4:5:6::8 1:2:3:4:5:6::8
# 1::7:8  1:2:3:4:5::7:8   1:2:3:4:5::8
# 1::6:7:8    1:2:3:4::6:7:8   1:2:3:4::8
# 1::5:6:7:8  1:2:3::5:6:7:8   1:2:3::8
# 1::4:5:6:7:8    1:2::4:5:6:7:8   1:2::8
# 1::3:4:5:6:7:8     1::3:4:5:6:7:8   1::8
# ::2:3:4:5:6:7:8    ::2:3:4:5:6:7:8  ::8       ::
# fe80::7:8%eth0     fe80::7:8%1  (link-local IPv6 addresses with zone index)
# ::255.255.255.255  ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped and IPv4-translated addresses)
# 2001:db8:3:4::192.0.2.33  64:ff9b::192.0.2.33 (IPv4-Embedded IPv6 Address)
ipv6groups = (
    r"(?:" + ipv6seg + r":){7,7}" + ipv6seg,
    r"(?:" + ipv6seg + r":){1,7}:",
    r"(?:" + ipv6seg + r":){1,6}:" + ipv6seg,
    r"(?:" + ipv6seg + r":){1,5}(?::" + ipv6seg + r"){1,2}",
    r"(?:" + ipv6seg + r":){1,4}(?::" + ipv6seg + r"){1,3}",
    r"(?:" + ipv6seg + r":){1,3}(?::" + ipv6seg + r"){1,4}",
    r"(?:" + ipv6seg + r":){1,2}(?::" + ipv6seg + r"){1,5}",
    ipv6seg + r":(?:(?::" + ipv6seg + r"){1,6})",
    r":(?:(?::" + ipv6seg + r"){1,7}|:)",
    r"fe80:(?::" + ipv6seg + r"){0,4}%[0-9a-zA-Z]{1,}",
    r"::(?:ffff(?::0{1,4}){0,1}:){0,1}[^\s:]" + ipv4addr,
    r"(?:" + ipv6seg + r":){1,6}:?[^\s:]" + ipv4addr,
)
ipv6addr = "|".join(["(?:{})".format(g) for g in ipv6groups[::-1]])

ipv6_pattern = (
    r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])(" + ipv6addr + r")(?:$|[\s@,?!;:'\"(.\p{Han}])"
)

ipv4_pattern = (
    r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])(" + ipv4addr + r")(?:$|[\s@,?!;:'\"(.\p{Han}])"
)


class DetectorKind(Enum):
    EMAIL = 1
    IPV4 = 2
    IPV6 = 3
    SECRET = 4
    REGEX = 5


class Detected(NamedTuple):
    kind: DetectorKind
    start: int
    end: int
    val: str


class BaseDetector(ABC):
    @abstractmethod
    def detect_all(self, content: str) -> list[Detected]:
        pass


class DetectorRegex(BaseDetector):
    def __init__(
        self,
        re_expression: re.Pattern,
        g: int = 1,
        kind: DetectorKind = DetectorKind.REGEX,
    ):
        self.re_expression = re_expression
        self.g = g
        self.kind = kind

    def finditer(self, content: str) -> Iterable[Detected]:
        matches = self.re_expression.finditer(content)
        for match in matches:
            # noinspection PyTypeChecker
            yield self._get_detected_from_match(match)

    def match(self, content: str) -> Optional[Detected]:
        if match := self.re_expression.match(content):
            return self._get_detected_from_match(match)

        return None

    def _get_detected_from_match(self, match: regex.Match) -> Detected:
        value = match.group(self.g)
        start, end = match.span(self.g)

        return Detected(
            kind=self.kind,
            start=start,
            end=end,
            val=value,
        )

    def detect_all(self, content: str) -> list[Detected]:
        return list(self.finditer(content))


class DetectorRegexEmail(BaseDetector):
    def __init__(self):
        re_expression = regex.compile(email_pattern, regex.MULTILINE | regex.VERBOSE)
        self.det = DetectorRegex(re_expression, kind=DetectorKind.EMAIL)

    def detect_all(self, content: str) -> list[Detected]:
        return self.det.detect_all(content)


class DetectorRegexIPV6(BaseDetector):
    def __init__(self):
        re_expression = regex.compile(ipv6_pattern, regex.MULTILINE | regex.VERBOSE)
        self.det = DetectorRegex(re_expression, kind=DetectorKind.IPV6)

    def detect_all(self, content: str) -> list[Detected]:
        return self.det.detect_all(content)


class DetectorRegexIPV4(BaseDetector):
    def __init__(self):
        re_expression = regex.compile(ipv4_pattern, regex.MULTILINE | regex.VERBOSE)
        self.det = DetectorRegex(re_expression, kind=DetectorKind.IPV4)

    def detect_all(self, content: str) -> list[Detected]:
        return self.det.detect_all(content)


class DetectorBasicAuthSecrets(BaseDetector):
    plugins = [{"name": "BasicAuthDetector"}]
    kind = DetectorKind.SECRET

    def _get_detected_from_secret(self, content: str, secret: PotentialSecret):
        start = content.index(f":{secret.secret_value}@") + 1
        end = start + len(secret.secret_value)

        return Detected(
            kind=self.kind,
            start=start,
            end=end,
            val=secret.secret_value,
        )

    def detect_all(self, content: str) -> list[Detected]:
        detected = []
        for secret in _run_detect_secrets_plugins(content, self.plugins):
            detected.append(self._get_detected_from_secret(content, secret))

        return detected


class DetectorTokenSecrets(BaseDetector):
    kind = DetectorKind.SECRET
    filters = [
        {"path": "detect_secrets.filters.heuristic.is_potential_uuid"},
        {"path": "detect_secrets.filters.heuristic.is_likely_id_string"},
        {"path": "detect_secrets.filters.heuristic.is_templated_secret"},
        {"path": "detect_secrets.filters.heuristic.is_sequential_string"},
    ]
    plugins = [
        {"name": "TwilioKeyDetector"},
        {"name": "JwtTokenDetector"},
        {"name": "ArtifactoryDetector"},
        {"name": "SendGridDetector"},
        {"name": "AzureStorageKeyDetector"},
        {"name": "DiscordBotTokenDetector"},
    ]

    def _get_detected_from_secret(
        self, content: str, secret: PotentialSecret
    ) -> Detected:
        start = content.index(secret.secret_value)
        end = start + len(secret.secret_value)

        return Detected(
            kind=self.kind,
            start=start,
            end=end,
            val=secret.secret_value,
        )

    def detect_all(self, content) -> list[Detected]:
        detected = []
        for secret in _run_detect_secrets_plugins(content, self.plugins):
            detected.append(self._get_detected_from_secret(content, secret))

        return detected


class DetectorKeywordsSecrets(BaseDetector):
    def __init__(self):
        self.detectors = []
        groups = [
            # we can add more groups when start analyzing file extensions
            # use case we're able to detect now. URL:
            # https://github.com/Yelp/detect-secrets/blob/master/tests/plugins/keyword_test.py#L126
            QUOTES_REQUIRED_DENYLIST_REGEX_TO_GROUP,
        ]
        for group in groups:
            for exp, g in group.items():
                self.detectors.append(DetectorRegex(exp, g, DetectorKind.SECRET))

    def detect_all(self, content: str) -> list[Detected]:
        detected = set()
        for det in self.detectors:
            detected.update(det.detect_all(content))
        return list(detected)


def _run_detect_secrets_plugins(
    content: str,
    plugins: list[dict[str, str]],
    filters: Optional[list[dict[str, str]]] = None,
) -> Iterable[PotentialSecret]:
    # https://github.com/Yelp/detect-secrets/blob/master/docs
    settings = {
        "plugins_used": plugins,
        "filters_used": filters if filters else [],
    }

    lines = content.splitlines()
    with transient_settings(settings) as _:
        lines_enumerated = list(enumerate(lines, start=1))

        # we use the private method of `detect_secrets` to avoid writing files with content
        # we do not need to pass any file names and can pass the empty value
        for potential_secret in _process_line_based_plugins(lines_enumerated, ""):
            yield potential_secret
