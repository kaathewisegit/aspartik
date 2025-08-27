from pdoc.doc import Doc, Module, Class, Function, Variable
from pathlib import Path
from typing import List, Dict
import json


def dump_json(module_name: str):
    module = Module.from_name(module_name)

    with open("target/pdoc.json", "w") as file:
        file.write(json.dumps(module_to_obj(module), indent=4))


def _vars(variables: List[Variable]) -> List[Dict]:
    return [variable_to_obj(var) for var in variables]


def _funcs(functions: List[Function]) -> List[Dict]:
    return [function_to_obj(func) for func in functions]


def module_to_obj(module: Module):
    return {
        **_common_items(module),
        "submodules": [module_to_obj(mod) for mod in module.submodules],
        "classes": [class_to_obj(cls) for cls in module.classes],
        "functions": _funcs(module.functions),
        "variables": _vars(module.variables),
    }


def class_to_obj(cls: Class):
    return {
        **_common_items(cls),
        "bases": cls.bases,
        "decorators": cls.decorators,
        "class_variables": _vars(cls.class_variables),
        "instance_variables": _vars(cls.instance_variables),
        "classmethods": _funcs(cls.classmethods),
        "staticmethods": _funcs(cls.staticmethods),
        "methods": _funcs(cls.methods),
    }


def function_to_obj(function: Function):
    return {
        **_common_items(function),
        "classmethod": function.is_classmethod,
        "staticmethod": function.is_staticmethod,
        "decorators": function.decorators,
        "def": function.funcdef,
        "signature": str(function.signature),
        "signature_without_self": str(function.signature_without_self),
    }


def variable_to_obj(variable: Variable):
    return {
        **_common_items(variable),
        "default": variable.default_value_str,
    }


def _common_items(value: Doc) -> dict:
    path = value.source_file
    if isinstance(path, str) and path != "<string>":
        path = str(Path(path).relative_to(Path.cwd()))
    else:
        path = None

    return {
        "type": value.kind,
        "fullname": value.fullname,
        "name": value.name,
        "docstring": value.docstring if value.docstring else None,
        "source": value.source if value.source else None,
        "source_lines": value.source_lines,
        "source_file": path,
    }
