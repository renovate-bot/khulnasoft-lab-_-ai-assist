import regex
from abc import ABC, abstractmethod
from typing import NamedTuple, Iterable, Optional

__all__ = [
    "Detected",
    "BaseDetector",
    "DetectorRegexEmail",
]

email_pattern = r'''
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
'''


class Detected(NamedTuple):
    start: int
    end: int
    val: str


class BaseDetector(ABC):
    @abstractmethod
    def detect_all(self, content: str) -> list[Detected]:
        pass


class DetectorRegex(BaseDetector):
    def __init__(self, pattern: str, flags: int = 0):
        self.re_expression = regex.compile(pattern, flags)

    def finditer(self, content: str) -> Iterable[Detected]:
        matches = self.re_expression.finditer(content)
        for match in matches:
            # noinspection PyUnresolvedReferences
            value = match.group(1)
            # noinspection PyUnresolvedReferences
            start, end = match.span(1)

            yield Detected(
                start=start,
                end=end,
                val=value,
            )

    def match(self, content: str) -> Optional[Detected]:
        if self.re_expression.match(content):
            # noinspection PyUnresolvedReferences
            value = match.group(1)
            # noinspection PyUnresolvedReferences
            start, end = match.span(1)

            return Detected(
                start=start,
                end=end,
                val=value,
            )

        return None

    def detect_all(self, content: str) -> list[Detected]:
        return list(self.finditer(content))


class DetectorRegexEmail(DetectorRegex):

    def __init__(self):
        super().__init__(email_pattern, flags=regex.MULTILINE | regex.VERBOSE)
