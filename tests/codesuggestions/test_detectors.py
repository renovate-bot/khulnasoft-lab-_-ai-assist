import pytest

from codesuggestions.suggestions.detectors import (
    DetectorRegexEmail, DetectorRegexIPV6, DetectorRegexIPV4, DetectorSecrets, Detected, DetectorKind
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


@pytest.mark.parametrize(
    "test_content,expected_output", [
        ("no secrets", []),
        ("basic auth: git clone https://username:1eeccr334f@gitlab.com/username/repository.git", [
            Detected(kind=DetectorKind.SECRET, start=39, end=49, val='1eeccr334f')
        ]),
        ("jwt with no signature: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
         ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ", [
             Detected(kind=DetectorKind.SECRET,start=23,end=134,val="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                                                                    ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI"
                                                                    "6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ")
         ]),
        ("artifactory credentials artif-key:AKCxxxxxxxxx1\nartifactoryx:_password=AKCxxxxxxxxx1", [
            Detected(kind=DetectorKind.SECRET, start=33, end=47, val=":AKCxxxxxxxxx1"),
            Detected(kind=DetectorKind.SECRET, start=70, end=84, val="=AKCxxxxxxxxx1"),
        ]),
        ("sendgrid tokens: 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", [
            Detected(kind=DetectorKind.SECRET, start=17, end=86, val="SG.ngeVfQFYQlKU0ufo8x5d1A.TwL2iGABf9DHoTf"
                                                                     "-09kqeF8tAmbihYzrnopKc-1s5cr")
        ]),
        ("azure: AccountKey=lJzRc1YdHaAA2KCNJJ1tkYwF/+mKK6Ygw0NGe170Xu592euJv2wYUtBlV8z+qnlcNQSnIYVTkLWntUO1F8j8rQ==", [
            Detected(kind=DetectorKind.SECRET, start=7, end=106, val="AccountKey=lJzRc1YdHaAA2KCNJJ1tkYwF/+"
                                                                     "mKK6Ygw0NGe170Xu592euJv2wYUtBlV8z+qnl"
                                                                     "cNQSnIYVTkLWntUO1F8j8rQ==")
        ]),
        ("discord: MTk4NjIyNDgzNDcxOTI1MjQ4.Cl2FMQ.ZnCjm1XVW7vRze4b7Cq4se7kKWs", [
            Detected(kind=DetectorKind.SECRET, start=9, end=68, val="MTk4NjIyNDgzNDcxOTI1MjQ4.Cl2FMQ"
                                                                    ".ZnCjm1XVW7vRze4b7Cq4se7kKWs")
        ]),
        ("twilio: SKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1\nACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1", [
            Detected(kind=DetectorKind.SECRET, start=8, end=42, val='SKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1'),
            Detected(kind=DetectorKind.SECRET, start=43, end=77, val='ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1')
        ])
    ]
)
def test_detector_secrets_detect_all(test_content, expected_output):
    det = DetectorSecrets()
    detected = det.detect_all(test_content)

    assert detected == expected_output
