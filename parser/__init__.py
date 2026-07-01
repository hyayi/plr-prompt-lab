"""parser/ — pluggable query-parser implementations.

Importing this package registers all bundled parser providers with the
registry so callers can use ``get_provider('parser')`` without having to
know which concrete class to instantiate.

Currently registered:
  - YamlParser  (version "qp_v0.4", slot "parser")
"""

from parser.yaml_parser import YamlParser  # noqa: F401 — side-effect: registers

__all__ = ["YamlParser"]
