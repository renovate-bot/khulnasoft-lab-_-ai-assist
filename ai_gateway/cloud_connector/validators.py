from ai_gateway.cloud_connector.user import UserClaims

X_GITLAB_DUO_SEAT_COUNT_HEADER = "X-Gitlab-Duo-Seat-Count"


def validate_duo_seat_count_header(
    claims: UserClaims, duo_seat_count_header: str
) -> str | None:
    # This indicates that an older instance is making a request. We remain backward compatible.
    if not claims.duo_seat_count:
        return None

    # Seat claim is present, but the header is missing. This should never happen.
    if not duo_seat_count_header:
        return f"Header is missing: '{X_GITLAB_DUO_SEAT_COUNT_HEADER}'"

    if claims.duo_seat_count != duo_seat_count_header:
        return f"Header mismatch '{X_GITLAB_DUO_SEAT_COUNT_HEADER}'"

    return None
