import pytest
import yaml

from eval_card_registry.lib.seed_io import safe_load_yaml


def test_safe_load_yaml_keeps_safe_loader_semantics():
    assert safe_load_yaml("items:\n  - id: example\n") == {
        "items": [{"id": "example"}]
    }
    with pytest.raises(yaml.constructor.ConstructorError):
        safe_load_yaml("!!python/object/apply:builtins.str [unsafe]")
