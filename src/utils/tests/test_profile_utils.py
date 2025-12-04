"""
Unit tests for src/utils/profile_utils.py
"""
import types
from src.utils import profile_utils

def test_iter_modules_returns_modules():
    """
    Test that iter_modules yields module names as strings.
    """
    modules = list(profile_utils.iter_modules("src"))
    assert all(isinstance(m, str) for m in modules)


def test_get_functions_filters_private():
    """
    Test that get_functions only returns public functions.
    """
    def public_func(): pass
    def _private_func(): pass
    dummy_module = types.ModuleType("dummy")
    setattr(dummy_module, "public_func", public_func)
    setattr(dummy_module, "_private_func", _private_func)
    funcs = dict(profile_utils.get_functions(dummy_module))
    assert "public_func" in funcs
    assert "_private_func" not in funcs


def test_fake_arg_types():
    """
    Test fake_arg returns correct dummy values for type hints and names.
    """
    import inspect
    def func(a: int, b: float, c: str, d: bool, e: list, f: dict, g: tuple, h, path=None, file=None): pass
    sig = inspect.signature(func)
    results = [profile_utils.fake_arg(param) for param in sig.parameters.values()]
    assert results[:7] == [1, 1.0, "test", True, [], {}, ()]
    assert results[8] == "dummy/path"
    assert results[9] == "dummy.txt"
