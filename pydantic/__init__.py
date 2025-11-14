from __future__ import annotations

from typing import Any, Dict


class FieldInfo:
    def __init__(
        self,
        default: Any = None,
        alias: str | None = None,
        description: str | None = None,
        default_factory: Any | None = None,
    ) -> None:
        self.default = default
        self.alias = alias
        self.description = description
        self.default_factory = default_factory


def Field(
    *,
    default: Any = None,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Any | None = None,
) -> FieldInfo:
    return FieldInfo(default=default, alias=alias, description=description, default_factory=default_factory)


class BaseModelMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: Dict[str, Any]):
        field_defaults: Dict[str, Any] = {}
        for key, value in list(namespace.items()):
            if isinstance(value, FieldInfo):
                field_defaults[key] = value
                namespace[key] = value.default
        cls = super().__new__(mcls, name, bases, namespace)
        setattr(cls, "_field_defaults", field_defaults)
        return cls


class BaseModel(metaclass=BaseModelMeta):
    def __init__(self, **data: Any) -> None:
        annotations = getattr(self, "__annotations__", {})
        defaults = getattr(self, "_field_defaults", {})
        for field in annotations:
            if field in data:
                value = data[field]
            elif field in defaults:
                info: FieldInfo = defaults[field]
                if info.default_factory is not None:
                    value = info.default_factory()
                else:
                    value = info.default
            else:
                value = getattr(self.__class__, field, None)
            annotation = annotations.get(field)
            if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, dict):
                value = annotation(**value)
            setattr(self, field, value)

    def model_dump(self) -> Dict[str, Any]:
        annotations = getattr(self, "__annotations__", {})
        result: Dict[str, Any] = {}
        for field in annotations:
            value = getattr(self, field)
            if isinstance(value, BaseModel):
                result[field] = value.model_dump()
            elif isinstance(value, list):
                result[field] = [item.model_dump() if isinstance(item, BaseModel) else item for item in value]
            else:
                result[field] = value
        return result

    def dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def model_copy(self, *, update: Dict[str, Any] | None = None):
        data = self.model_dump()
        if update:
            data.update(update)
        return self.__class__(**data)


__all__ = ["BaseModel", "Field"]
