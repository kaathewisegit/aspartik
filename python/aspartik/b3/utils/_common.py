def raise_import(name: str):
    raise ImportError(
        f"Library `{name}` not found.  Install aspartik with the `analysis` feature to use this module"
    )
