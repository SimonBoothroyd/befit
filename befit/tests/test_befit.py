import importlib


def test_befit():
    assert importlib.import_module("befit") is not None
