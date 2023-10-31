import typing

import starlette.authentication
from starlette.authentication import requires  # noqa: F401
from starlette.requests import HTTPConnection


def has_required_scope(conn: HTTPConnection, scopes: typing.Sequence[str]) -> bool:
    if conn.user.is_debug is True:
        return True

    for scope in scopes:
        sub_scopes = scope.split("|")
        inside_scopes = [scope in conn.auth.scopes for scope in sub_scopes]
        if not any(inside_scopes):
            return False
    return True


starlette.authentication.has_required_scope = has_required_scope
