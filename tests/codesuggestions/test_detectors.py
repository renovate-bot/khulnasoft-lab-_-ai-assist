import pytest

from codesuggestions.suggestions.detectors import (
    DetectorRegexEmail, DetectorRegexIPV6, DetectorRegexIPV4, Detected, DetectorKind
)


@pytest.mark.parametrize(
    "test_content,expected_output", [
        ("no address", []),
        ("one address: email@ex.com", [Detected(kind=DetectorKind.EMAIL, start=13, end=25, val="email@ex.com")]),
        ("one: one@ex.com, two: two@ex.com",
         [Detected(kind=DetectorKind.EMAIL, start=5, end=15, val="one@ex.com"),
          Detected(kind=DetectorKind.EMAIL, start=22, end=32, val="two@ex.com")]),
        ("wrong address: email@com", []),
        ("wrong address: email@.com", [])
    ]
)
def test_detector_email_detect_all(test_content, expected_output):
    det = DetectorRegexEmail()
    detected = det.detect_all(test_content)

    assert detected == expected_output


@pytest.mark.parametrize(
    "test_content,expected_output", [
        ("no ip address", []),
        ("no ipv6 address 33.01.33.33", []),
        ("test 1:2:3:4:5:6:7:8", [Detected(kind=DetectorKind.IPV6, start=5, end=20, val="1:2:3:4:5:6:7:8")]),
        ("test 1::, 1:2:3:4:5:6:7::", [
            Detected(kind=DetectorKind.IPV6, start=5, end=8, val="1::"),
            Detected(kind=DetectorKind.IPV6, start=10, end=25, val="1:2:3:4:5:6:7::")
        ]),
        ("test 1::8, 1:2:3:4:5:6::8, 1:2:3:4:5:6::8", [
            Detected(kind=DetectorKind.IPV6, start=5, end=9, val="1::8"),
            Detected(kind=DetectorKind.IPV6, start=11, end=25, val="1:2:3:4:5:6::8"),
            Detected(kind=DetectorKind.IPV6, start=27, end=41, val="1:2:3:4:5:6::8")
        ]),
        ("test 1::7:8, 1:2:3:4:5::7:8, 1:2:3:4:5::8", [
            Detected(kind=DetectorKind.IPV6, start=5, end=11, val="1::7:8"),
            Detected(kind=DetectorKind.IPV6, start=13, end=27, val="1:2:3:4:5::7:8"),
            Detected(kind=DetectorKind.IPV6, start=29, end=41, val="1:2:3:4:5::8")
        ]),
        ("test 1::6:7:8, 1:2:3:4::6:7:8, 1:2:3:4::8", [
            Detected(kind=DetectorKind.IPV6, start=5, end=13, val="1::6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=15, end=29, val="1:2:3:4::6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=31, end=41, val="1:2:3:4::8")
        ]),
        ("test 1::5:6:7:8, 1:2:3::5:6:7:8, 1:2:3::8", [
            Detected(kind=DetectorKind.IPV6, start=5, end=15, val="1::5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=17, end=31, val="1:2:3::5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=33, end=41, val="1:2:3::8")
        ]),
        ("test 1::4:5:6:7:8, 1:2::4:5:6:7:8, 1:2::8", [
            Detected(kind=DetectorKind.IPV6, start=5, end=17, val="1::4:5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=19, end=33, val="1:2::4:5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=35, end=41, val="1:2::8")
        ]),
        ("test 1::3:4:5:6:7:8, 1::3:4:5:6:7:8, 1::8", [
            Detected(kind=DetectorKind.IPV6, start=5, end=19, val="1::3:4:5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=21, end=35, val="1::3:4:5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=37, end=41, val="1::8")
        ]),
        ("test ::2:3:4:5:6:7:8, ::2:3:4:5:6:7:8, ::8, ::", [
            Detected(kind=DetectorKind.IPV6, start=5, end=20, val="::2:3:4:5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=22, end=37, val="::2:3:4:5:6:7:8"),
            Detected(kind=DetectorKind.IPV6, start=39, end=42, val="::8"),
            Detected(kind=DetectorKind.IPV6, start=44, end=46, val="::")
        ]),
        ("test fe80::7:8%eth0, fe80::7:8%1", [
            Detected(kind=DetectorKind.IPV6, start=5, end=19, val="fe80::7:8%eth0"),
            Detected(kind=DetectorKind.IPV6, start=21, end=32, val="fe80::7:8%1")
        ]),
        ("test ::255.255.255.255, ::ffff:255.255.255.255, ::ffff:0:255.255.255.255", [
            Detected(kind=DetectorKind.IPV6, start=5, end=22, val="::255.255.255.255"),
            Detected(kind=DetectorKind.IPV6, start=24, end=46, val="::ffff:255.255.255.255"),
            Detected(kind=DetectorKind.IPV6, start=48, end=72, val="::ffff:0:255.255.255.255")
        ]),
        ("test 2001:db8:3:4::192.0.2.33, 64:ff9b::192.0.2.33", [
            Detected(kind=DetectorKind.IPV6, start=5, end=29, val="2001:db8:3:4::192.0.2.33"),
            Detected(kind=DetectorKind.IPV6, start=31, end=50, val="64:ff9b::192.0.2.33")
        ]),
    ]
)
def test_detector_ipv6_detect_all(test_content, expected_output):
    det = DetectorRegexIPV6()
    detected = det.detect_all(test_content)

    assert detected == expected_output


@pytest.mark.parametrize(
    "test_content,expected_output", [
        ("test no ip", []),
        ("test no ip 2020.10", []),
        ("test no ip 20.10.01", []),
        ("test no ipv4 1::3:4:5:6:7:8", []),
        ("test 127.0.0.1", [Detected(kind=DetectorKind.IPV4, start=5, end=14, val='127.0.0.1')]),
        ("test 255.255.255.255", [Detected(kind=DetectorKind.IPV4, start=5, end=20, val='255.255.255.255')]),
        ("test 10.1.1.124", [Detected(kind=DetectorKind.IPV4, start=5, end=15, val='10.1.1.124')]),
        ("test 10.01.1.124", [Detected(kind=DetectorKind.IPV4, start=5, end=16, val='10.01.1.124')]),  # detect this ip even if it's invalid
    ]
)
def test_detector_ipv4_detect_all(test_content, expected_output):
    det = DetectorRegexIPV4()
    detected = det.detect_all(test_content)

    assert detected == expected_output
