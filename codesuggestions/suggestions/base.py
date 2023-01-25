from codesuggestions.models import BaseModel
from codesuggestions.suggestions.detectors import (
    DetectorRegexEmail,
    DetectorRegexIPV4,
    DetectorRegexIPV6,
    DetectorBasicAuthSecrets,
    DetectorTokenSecrets,
    Detected,
    DetectorKind,
)

__all__ = [
    "DEFAULT_REPLACEMENT_EMAIL",
    "DEFAULT_REPLACEMENT_IPV4",
    "DEFAULT_REPLACEMENT_IPV6",
    "DEFAULT_REPLACEMENT_SECRET",
    "CodeSuggestionsUseCase",
]

DEFAULT_REPLACEMENT_EMAIL = "<email placeholder|email@example.com>"
DEFAULT_REPLACEMENT_IPV4 = "<ipv4 placeholder|x.x.x.x>"
DEFAULT_REPLACEMENT_IPV6 = "<ipv6 placeholder|x:x:x:x:x:x:x:x>"
DEFAULT_REPLACEMENT_SECRET = "<secret placeholder|secret>"


class CodeSuggestionsUseCase:
    def __init__(self, model: BaseModel):
        self.model = model
        self.detectors = [
            DetectorRegexEmail(),
            # a different order of IPVx detectors may change the output
            DetectorRegexIPV6(),
            DetectorRegexIPV4(),
            DetectorBasicAuthSecrets(),
            DetectorTokenSecrets(),
        ]
        self.pii_replacements = {
            DetectorKind.EMAIL: DEFAULT_REPLACEMENT_EMAIL,
            DetectorKind.IPV4: DEFAULT_REPLACEMENT_IPV4,
            DetectorKind.IPV6: DEFAULT_REPLACEMENT_IPV6,
            DetectorKind.SECRET: DEFAULT_REPLACEMENT_SECRET,
        }

    def _detect_pii(self, content: str) -> list[Detected]:
        detected = []
        for detector in self.detectors:
            detected.extend(detector.detect_all(content))

        return detected

    def _redact_pii(self, content: str, detected: list[Detected]):
        detected = sorted(detected, key=lambda x: (x.start, -x.end))

        step = 0
        subparts = []
        for d in detected:
            if step > d.start:
                # skip: the previous detection overlaps the current one
                continue
            if subtext := content[step: d.start]:
                subparts.append(subtext)

            subparts.append(self.pii_replacements[d.kind])
            step = d.end

        # add remaining content
        if step < len(content):
            subparts.append(content[step:])

        redacted = "".join(subparts)

        return redacted

    def _postprocess_pii(self, completion: str) -> str:
        pii_detected = self._detect_pii(completion)
        return self._redact_pii(completion, pii_detected)

    def __call__(self, prompt: str) -> str:
        completion = self.model(prompt)

        completion = _process_content(completion, self._postprocess_pii)

        return completion


def _process_content(content: str, *funcs) -> str:
    for func in funcs:
        content = func(content)
    return content
