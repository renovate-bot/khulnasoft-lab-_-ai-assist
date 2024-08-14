from starlette.authentication import AuthenticationError

X_GITLAB_DUO_SEAT_COUNT_HEADER = "X-Gitlab-Duo-Seat-Count"


def validate_duo_seat_count_header(claims, headers):
    # This indicates that an older instance is making a request. We remain backward compatible.
    if not claims.duo_seat_count:
        return

    duo_seat_count_header = headers.get(X_GITLAB_DUO_SEAT_COUNT_HEADER)

    # Seat claim is present, but the header is missing. This should never happen.
    if not duo_seat_count_header:
        raise AuthenticationError(
            f"Header is missing: '{X_GITLAB_DUO_SEAT_COUNT_HEADER}'"
        )

    if claims.duo_seat_count != duo_seat_count_header:
        raise AuthenticationError(f"Header mismatch '{X_GITLAB_DUO_SEAT_COUNT_HEADER}'")
