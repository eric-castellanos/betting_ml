import importlib
import inspect
import pkgutil
import pyinstrument


# ---------------------------------------------------------
# Module discovery
# ---------------------------------------------------------
def iter_modules(package_root="src"):
    package = importlib.import_module(package_root)
    package_path = package.__file__.replace("__init__.py", "")
    
    for module_info in pkgutil.walk_packages([package_path], prefix=f"{package_root}."):
        yield module_info.name


# ---------------------------------------------------------
# Function discovery
# ---------------------------------------------------------
def get_functions(module):
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        yield name, obj


# ---------------------------------------------------------
# Fake argument generation using annotations or defaults
# ---------------------------------------------------------
def fake_arg(param):
    """Generate a dummy argument based on type hints or name."""
    annotation = param.annotation

    # Typed parameters
    if annotation == int:
        return 1
    if annotation == float:
        return 1.0
    if annotation == str:
        return "test"
    if annotation == bool:
        return True
    if annotation == list:
        return []
    if annotation == dict:
        return {}
    if annotation == tuple:
        return ()
    
    # Unannotated but common names
    lname = param.name.lower()
    if "path" in lname:
        return "dummy/path"
    if "file" in lname:
        return "dummy.txt"

    # Generic fallback
    return None


def build_args(func):
    """Builds usable positional args for a function."""
    sig = inspect.signature(func)
    args = []
    for name, p in sig.parameters.items():
        # Skip *args or **kwargs
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Use default if available
        if p.default is not inspect._empty:
            args.append(p.default)
            continue

        # Otherwise generate fake value
        args.append(fake_arg(p))

    return args


# ---------------------------------------------------------
# Profiling logic with generated args
# ---------------------------------------------------------
def profile_function(module_name, func_name, func):
    args = build_args(func)

    profiler = pyinstrument.Profiler()

    try:
        profiler.start()
        func(*args)
        profiler.stop()

        safe = f"{module_name}_{func_name}".replace(".", "_")
        profiler.write_html(f"pyinstrument_{safe}.html")

        print(f" → Profiled with args={args}")

    except Exception as e:
        profiler.stop()
        print(f"Function {module_name}.{func_name} failed during profiling: {e}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    for module_name in iter_modules():
        print(f"\nImporting module: {module_name}")
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"Skipping module due to import error: {module_name}: {e}")
            continue

        for func_name, func in get_functions(module):
            print(f"Profiling {module_name}.{func_name}")
            profile_function(module_name, func_name, func)


if __name__ == "__main__":
    main()

