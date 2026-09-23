"""域注册表：按 role 名称获取 DomainSpec。"""

from __future__ import annotations

from listen2serve.domains.base import DomainSpec

_REGISTRY: dict[str, DomainSpec] = {}


def get_domain(role: str) -> DomainSpec:
    """获取域定义（懒加载注册）。"""
    if role not in _REGISTRY:
        if role == "collection":
            from listen2serve.domains.collection import build_domain
        elif role == "marketing":
            from listen2serve.domains.marketing import build_domain
        elif role == "hotline":
            from listen2serve.domains.hotline import build_domain
        else:
            raise ValueError(f"未知域: {role}")
        _REGISTRY[role] = build_domain()
    return _REGISTRY[role]


def available_roles() -> list[str]:
    return ["collection", "marketing", "hotline"]
