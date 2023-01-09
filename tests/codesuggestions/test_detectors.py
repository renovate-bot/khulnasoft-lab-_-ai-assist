import pytest

from codesuggestions.suggestions.detectors import DetectorRegexEmail, Detected


@pytest.mark.parametrize(
    "test_content,expected_output", [
        ("no address", []),
        ("one address: email@ex.com", [Detected(start=13, end=25, val="email@ex.com")]),
        ("one: one@ex.com, two: two@ex.com",
         [Detected(start=5, end=15, val="one@ex.com"),
          Detected(start=22, end=32, val="two@ex.com")]),
        ("wrong address: email@com", []),
        ("wrong address: email@.com", [])
    ]
)
def test_detector_email_detect_all(test_content, expected_output):
    det = DetectorRegexEmail()
    detected = det.detect_all(test_content)

    assert detected == expected_output
