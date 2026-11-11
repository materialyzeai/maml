"""Inspect function args."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def get_full_args(func: Callable) -> list:
    """
    Get args from function.

    Args:
        func (callable): function to determine the args
    """
    all_params = list(signature(func).parameters.keys())
    return [i for i in all_params if i != "self"]


def get_param_types(func):
    """
    Get param and type info.

    Args:
        func (callable): function to determine the arg types
    """
    all_params = get_full_args(func)
    params = signature(func).parameters
    param_dict = {}

    for i in all_params:
        annotation = params[i].annotation
        if isinstance(annotation, str):
            # Postponed (PEP 563) annotations are plain strings, e.g. "Structure | None".
            param_dict[i] = annotation.split("|")[0].strip()
            continue
        param_type = getattr(annotation, "__args__", None)
        if param_type is not None:
            param_dict[i] = param_type[0].__name__
        else:
            param_dict[i] = annotation.__name__
    return param_dict
