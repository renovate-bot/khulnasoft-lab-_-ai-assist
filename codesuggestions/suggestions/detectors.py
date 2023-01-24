import regex
from abc import ABC, abstractmethod
from typing import NamedTuple, Iterable, Optional
from enum import Enum

# noinspection PyProtectedMember
from detect_secrets.core.scan import _process_line_based_plugins, PotentialSecret
from detect_secrets.settings import transient_settings

__all__ = [
    "Detected",
    "DetectorKind",
    "BaseDetector",
    "DetectorRegexEmail",
    "DetectorRegexIPV6",
    "DetectorRegexIPV4",
    "DetectorSecrets",
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

ipv4seg = r'(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
ipv4addr = r'(?:(?:' + ipv4seg + r'\.){3,3}' + ipv4seg + r')'
ipv6seg = r'(?:(?:[0-9a-fA-F]){1,4})'

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
    r'(?:' + ipv6seg + r':){7,7}' + ipv6seg,
    r'(?:' + ipv6seg + r':){1,7}:',
    r'(?:' + ipv6seg + r':){1,6}:' + ipv6seg,
    r'(?:' + ipv6seg + r':){1,5}(?::' + ipv6seg + r'){1,2}',
    r'(?:' + ipv6seg + r':){1,4}(?::' + ipv6seg + r'){1,3}',
    r'(?:' + ipv6seg + r':){1,3}(?::' + ipv6seg + r'){1,4}',
    r'(?:' + ipv6seg + r':){1,2}(?::' + ipv6seg + r'){1,5}',
    ipv6seg + r':(?:(?::' + ipv6seg + r'){1,6})',
    r':(?:(?::' + ipv6seg + r'){1,7}|:)',
    r'fe80:(?::' + ipv6seg + r'){0,4}%[0-9a-zA-Z]{1,}',
    r'::(?:ffff(?::0{1,4}){0,1}:){0,1}[^\s:]' + ipv4addr,
    r'(?:' + ipv6seg + r':){1,6}:?[^\s:]' + ipv4addr,
)
ipv6addr = '|'.join(['(?:{})'.format(g) for g in ipv6groups[::-1]])

ipv6_pattern = (
    r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])("
    + ipv6addr
    + r")(?:$|[\s@,?!;:'\"(.\p{Han}])"
)

ipv4_pattern = (
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])("
        + ipv4addr
        + r")(?:$|[\s@,?!;:'\"(.\p{Han}])"
)

detector_secrets_filters = [
    # https://github.com/Yelp/detect-secrets/blob/master/docs/filters.md#built-in-filters
    {"path": "detect_secrets.filters.heuristic.is_potential_uuid"},
    {"path": "detect_secrets.filters.heuristic.is_likely_id_string"},
    {"path": "detect_secrets.filters.heuristic.is_templated_secret"},
    {"path": "detect_secrets.filters.heuristic.is_sequential_string"},
]
detector_secrets_plugins = [
    {"name": "TwilioKeyDetector"},
    {"name": "JwtTokenDetector"},
    {"name": "BasicAuthDetector"},
    {"name": "ArtifactoryDetector"},
    {"name": "SendGridDetector"},
    {"name": "AzureStorageKeyDetector"},
    {"name": "DiscordBotTokenDetector"},
]


class DetectorKind(Enum):
    EMAIL = 1
    IPV4 = 2
    IPV6 = 3
    SECRET = 4


class Detected(NamedTuple):
    kind: DetectorKind
    start: int
    end: int
    val: str


class BaseDetector(ABC):
    @abstractmethod
    def detect_all(self, content: str) -> list[Detected]:
        pass

    @property
    @abstractmethod
    def kind(self):
        pass


class DetectorRegex(BaseDetector, ABC):
    def __init__(self, pattern: str, flags: int = 0):
        self.re_expression = regex.compile(pattern, flags)

    def finditer(self, content: str) -> Iterable[Detected]:
        matches = self.re_expression.finditer(content)
        for match in matches:
            # noinspection PyTypeChecker
            yield self._get_detected_from_match(match)

    def match(self, content: str) -> Optional[Detected]:
        if match := self.re_expression.match(content):
            return self._get_detected_from_match(match)

        return None

    def _get_detected_from_match(self, match: regex.Match, g: int = 1) -> Detected:
        value = match.group(g)
        start, end = match.span(g)

        return Detected(
            kind=self.kind,
            start=start,
            end=end,
            val=value,
        )

    def detect_all(self, content: str) -> list[Detected]:
        return list(self.finditer(content))


class DetectorRegexEmail(DetectorRegex):

    def __init__(self):
        super().__init__(email_pattern, flags=regex.MULTILINE | regex.VERBOSE)

    @property
    def kind(self):
        return DetectorKind.EMAIL


class DetectorRegexIPV6(DetectorRegex):
    def __init__(self):
        super().__init__(ipv6_pattern, flags=regex.MULTILINE | regex.VERBOSE)

    @property
    def kind(self):
        return DetectorKind.IPV6


class DetectorRegexIPV4(DetectorRegex):
    def __init__(self):
        super().__init__(ipv4_pattern, flags=regex.MULTILINE | regex.VERBOSE)

    @property
    def kind(self):
        return DetectorKind.IPV4


class DetectorSecrets(BaseDetector):

    def _get_detected_from_secret(self, content: str, secret: PotentialSecret) -> Detected:
        start = content.index(secret.secret_value)
        end = start + len(secret.secret_value)

        return Detected(
            kind=self.kind,
            start=start,
            end=end,
            val=secret.secret_value,
        )

    def detect_all(self, content: str) -> list[Detected]:
        detected = []
        lines = content.splitlines()

        settings = {"plugins_used": detector_secrets_plugins, "filters_used": detector_secrets_filters}
        with transient_settings(settings) as _:
            lines_enumerated = list(enumerate(lines, start=1))

            # we use the private method of `detect_secrets` to avoid writing files with content
            # we do not need to pass any file names and can pass the empty value
            for potential_secret in _process_line_based_plugins(lines_enumerated, ""):
                detected_secret = self._get_detected_from_secret(content, potential_secret)
                detected.append(detected_secret)

        return detected

    @property
    def kind(self):
        return DetectorKind.SECRET
