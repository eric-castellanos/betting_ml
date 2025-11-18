import importlib
import inspect
import pkgutil
import pyinstrument

def iter_modules(package_root="src"):
    package = importlib.import_module(package_root)
    package_path = package.__file__.replace("__init__.py", "")
    
    for module_info in pkgutil.walk_packages([package_path], prefix=f"{package_root}."):
        yield module_info.name

def get_functions(module):
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Skip click wrappers, pytest functions, dunders
        if name.startswith("_"):
            continue
        yield name, obj

def profile_function(module_name, func_name, func):
    profiler = pyinstrument.Profiler()
    try:
        sig = inspect.signature(func)
        if any(
            p.default is inspect._empty and
            p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            for p in sig.parameters.values()
        ):
            print(f"Skipping {module_name}.{func_name} — requires args")
            return

        profiler.start()
        func()  # now safe
        profiler.stop()

        safe = f"{module_name}_{func_name}".replace(".", "_")
        profiler.write_html(f"pyinstrument_{safe}.html")

    except Exception as e:
        print(f"Function {module_name}.{func_name} failed: {e}")


def main():
    for module_name in iter_modules():
        print(f"Importing module: {module_name}")
        try:
            module = importlib.import_module(module_name)
        except Exception:
            print(f"Skipping module due to import error: {module_name}")
            continue

        for func_name, func in get_functions(module):
            print(f"Profiling {module_name}.{func_name}")
            profile_function(module_name, func_name, func)

if __name__ == "__main__":
    main()
