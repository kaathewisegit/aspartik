def raise_import(e: ModuleNotFoundError):
    raise ImportError(
        "Install aspartik with the `analysis` feature to use this module"
    ) from e
