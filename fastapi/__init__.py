from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str | Dict[str, Any]) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class status:
    HTTP_200_OK = 200
    HTTP_400_BAD_REQUEST = 400
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404


class Depends:
    def __init__(self, dependency: Callable[..., Any]) -> None:
        self.dependency = dependency


class Header:
    def __init__(self, default: Any | None = None, *, alias: str | None = None) -> None:
        self.default = default
        self.alias = alias


@dataclass
class _Route:
    method: str
    path: str
    endpoint: Callable[..., Any]


class APIRouter:
    def __init__(self) -> None:
        self.routes: List[_Route] = []

    def get(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("GET", path)

    def post(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("POST", path)

    def _register(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes.append(_Route(method=method, path=path, endpoint=func))
            return func

        return decorator


class FastAPI(APIRouter):
    def __init__(self, *, title: str, version: str) -> None:
        super().__init__()
        self.title = title
        self.version = version

    def include_router(self, router: APIRouter) -> None:
        self.routes.extend(router.routes)

    def find_route(self, method: str, path: str) -> _Route:
        for route in self.routes:
            if route.method == method and route.path == path:
                return route
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Route {method} {path} not found")

    async def __call__(self, scope, receive, send):  # pragma: no cover - exercised via real server
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            raise RuntimeError("Unsupported scope type")
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        method = scope["method"].upper()
        path = scope["path"]
        try:
            route = self.find_route(method, path)
            json_body = json.loads(body.decode("utf-8")) if body else {}
            headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
            payload = _call_with_dependencies(route.endpoint, json_body=json_body, headers=headers)
            response_data = _serialize_response(payload)
            status_code = status.HTTP_200_OK
        except HTTPException as exc:
            response_data = {"detail": exc.detail}
            status_code = exc.status_code
        response_body = json.dumps(response_data).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(response_body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": response_body})


def _call_with_dependencies(
    func: Callable[..., Any],
    *,
    json_body: Dict[str, Any],
    headers: Dict[str, str],
) -> Any:
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    signature = inspect.signature(func)
    type_hints = typing.get_type_hints(func)
    kwargs: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        annotation = type_hints.get(name, param.annotation)
        default = param.default
        if isinstance(default, Depends):
            kwargs[name] = _resolve_dependency(default.dependency, headers=normalized_headers)
        elif isinstance(default, Header):
            alias = (default.alias or name).lower()
            if alias not in normalized_headers:
                if default.default is not None:
                    kwargs[name] = default.default
                else:
                    raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Missing header '{alias}'")
            else:
                kwargs[name] = normalized_headers[alias]
        elif annotation is not inspect._empty and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            kwargs[name] = annotation(**json_body)
        else:
            if name in json_body:
                kwargs[name] = json_body[name]
            elif default is not inspect._empty:
                kwargs[name] = default
            else:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Missing body field '{name}'")
    return func(**kwargs)


def _resolve_dependency(func: Callable[..., Any], *, headers: Dict[str, str]) -> Any:
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    signature = inspect.signature(func)
    type_hints = typing.get_type_hints(func)
    kwargs: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        default = param.default
        if isinstance(default, Depends):
            kwargs[name] = _resolve_dependency(default.dependency, headers=normalized_headers)
        elif isinstance(default, Header):
            alias = (default.alias or name).lower()
            if alias not in normalized_headers:
                if default.default is not None:
                    kwargs[name] = default.default
                else:
                    raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Missing header '{alias}'")
            else:
                kwargs[name] = normalized_headers[alias]
        else:
            if default is not inspect._empty:
                kwargs[name] = default
            else:
                annotation = type_hints.get(name, param.annotation)
                if annotation is str:
                    kwargs[name] = ""
                else:
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Missing dependency parameter '{name}'")
    return func(**kwargs)


def _serialize_response(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict") and callable(result.dict):
        return result.dict()
    if isinstance(result, (list, tuple)):
        return [
            _serialize_response(item)
            for item in result
        ]
    if isinstance(result, dict):
        return result
    return result


class _Response:
    def __init__(self, status_code: int, data: Any):
        self.status_code = status_code
        self._data = data

    def json(self) -> Any:
        return self._data


class TestClient:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def get(self, path: str, *, headers: Optional[Dict[str, str]] = None) -> _Response:
        return self._request("GET", path, json_body={}, headers=headers or {})

    def post(
        self,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> _Response:
        return self._request("POST", path, json_body=json or {}, headers=headers or {})

    def _request(self, method: str, path: str, *, json_body: Dict[str, Any], headers: Dict[str, str]) -> _Response:
        try:
            route = self._app.find_route(method, path)
        except HTTPException as exc:
            return _Response(exc.status_code, {"detail": exc.detail})
        try:
            payload = _call_with_dependencies(route.endpoint, json_body=json_body, headers=headers)
            data = _serialize_response(payload)
            return _Response(status.HTTP_200_OK, data)
        except HTTPException as exc:
            return _Response(exc.status_code, {"detail": exc.detail})


__all__ = [
    "APIRouter",
    "FastAPI",
    "Depends",
    "Header",
    "HTTPException",
    "status",
    "TestClient",
]
