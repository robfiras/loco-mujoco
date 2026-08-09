"""Coverage for loco_mujoco.utils.dataset - the LOCOMUJOCO_VARIABLES manager.

Despite living under ``utils/dataset.py`` this module does no downloading: it is
pure YAML config + argparse CLI plumbing for the ``LOCOMUJOCO_*`` path
variables. It was almost entirely untested (12%). Everything here is exercised
against a throwaway YAML file by monkeypatching ``loco_mujoco.PATH_TO_VARIABLES``
(read at call-time) and, for the CLI entry points, ``sys.argv``.
"""
import sys

import pytest
import yaml

import loco_mujoco
from loco_mujoco.utils import dataset as ds


@pytest.fixture
def var_file(tmp_path, monkeypatch):
    """Point PATH_TO_VARIABLES at a throwaway file (not yet created)."""
    path = tmp_path / "LOCOMUJOCO_VARIABLES.yaml"
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(path))
    return path


def _load(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


# --------------------------------------------------------------------------- #
# add_variable / remove_variable - string variables
# --------------------------------------------------------------------------- #
def test_add_and_remove_str_variable(var_file):
    ds.add_variable("LOCOMUJOCO_AMASS_PATH", "/data/amass", quiet=True)
    assert _load(var_file)["LOCOMUJOCO_AMASS_PATH"] == "/data/amass"

    # overwrite
    ds.add_variable("LOCOMUJOCO_AMASS_PATH", "/data/amass2", quiet=True)
    assert _load(var_file)["LOCOMUJOCO_AMASS_PATH"] == "/data/amass2"

    # remove (str -> whole key deleted)
    ds.remove_variable("LOCOMUJOCO_AMASS_PATH")
    assert "LOCOMUJOCO_AMASS_PATH" not in _load(var_file)


# --------------------------------------------------------------------------- #
# add_variable / remove_variable - list variables
# --------------------------------------------------------------------------- #
def test_add_and_remove_list_variable(var_file):
    ds.add_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m1", quiet=True)
    ds.add_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m2", quiet=True)
    assert _load(var_file)["LOCOMUJOCO_CUSTOM_MODELS_PATH"] == ["/m1", "/m2"]

    # adding a duplicate is a no-op (covers the "already exists" branch)
    ds.add_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m1", quiet=True)
    assert _load(var_file)["LOCOMUJOCO_CUSTOM_MODELS_PATH"] == ["/m1", "/m2"]

    # remove one element -> key stays
    ds.remove_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m1")
    assert _load(var_file)["LOCOMUJOCO_CUSTOM_MODELS_PATH"] == ["/m2"]

    # remove the last element -> key removed entirely
    ds.remove_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m2")
    assert "LOCOMUJOCO_CUSTOM_MODELS_PATH" not in _load(var_file)


def test_add_variable_type_errors(var_file):
    # unknown variable name + no explicit type
    with pytest.raises(KeyError):
        ds.add_variable("SOME_UNKNOWN_VAR", "/x", quiet=True)

    # predefined variable with the wrong explicit type
    with pytest.raises(AssertionError):
        ds.add_variable("LOCOMUJOCO_AMASS_PATH", "/x", variable_type="list", quiet=True)

    # invalid type for a fresh variable
    with pytest.raises(ValueError):
        ds.add_variable("SOME_UNKNOWN_VAR", "/x", variable_type="dict", quiet=True)

    # non-string value
    with pytest.raises(AssertionError):
        ds.add_variable("LOCOMUJOCO_AMASS_PATH", 123, quiet=True)


def test_remove_variable_edge_cases(var_file):
    # file does not exist yet
    ds.remove_variable("LOCOMUJOCO_AMASS_PATH")  # prints + returns

    # create a file with a str + a list var
    ds.add_variable("LOCOMUJOCO_AMASS_PATH", "/a", quiet=True)
    ds.add_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m1", quiet=True)

    # removing a non-existent variable just returns
    ds.remove_variable("NOT_THERE")

    # removing from a list variable without a value is an error
    with pytest.raises(ValueError):
        ds.remove_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH")


def test_yaml_helper_edge_cases(var_file):
    # delete from a non-existent file
    ds._delete_path_in_yaml_conf("LOCOMUJOCO_AMASS_PATH", str(var_file))

    # create file, then delete a missing attr
    ds.add_variable("LOCOMUJOCO_AMASS_PATH", "/a", quiet=True)
    ds._delete_path_in_yaml_conf("NOPE", str(var_file))

    # remove-from-list helper on a missing file
    missing = str(var_file) + ".missing"
    ds._remove_path_from_yaml_list("/x", "LOCOMUJOCO_CUSTOM_MODELS_PATH", missing)

    # remove-from-list when attr missing / path missing
    ds._remove_path_from_yaml_list("/x", "LOCOMUJOCO_CUSTOM_MODELS_PATH", str(var_file))
    ds.add_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m1", quiet=True)
    ds._remove_path_from_yaml_list("/not-in-list", "LOCOMUJOCO_CUSTOM_MODELS_PATH", str(var_file))


# --------------------------------------------------------------------------- #
# CLI entry points (argparse via sys.argv)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func,var_key", [
    (ds.set_amass_path, "LOCOMUJOCO_AMASS_PATH"),
    (ds.set_smpl_model_path, "LOCOMUJOCO_SMPL_MODEL_PATH"),
    (ds.set_converted_amass_path, "LOCOMUJOCO_CONVERTED_AMASS_PATH"),
    (ds.set_converted_lafan1_path, "LOCOMUJOCO_CONVERTED_LAFAN1_PATH"),
    (ds.set_lafan1_path, "LOCOMUJOCO_LAFAN1_PATH"),
])
def test_simple_set_path_clis(var_file, monkeypatch, func, var_key):
    monkeypatch.setattr(sys, "argv", ["prog", "--path", "/some/path"])
    func()
    assert _load(var_file)[var_key] == "/some/path"


def test_set_all_caches_cli(var_file, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--path", "/caches"])
    ds.set_all_caches()
    data = _load(var_file)
    assert data["LOCOMUJOCO_CONVERTED_AMASS_PATH"].endswith("AMASS")
    assert data["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"].endswith("LAFAN1")
    assert data["LOCOMUJOCO_CONVERTED_DEFAULT_PATH"].endswith("DEFAULT")


def test_add_and_remove_variable_clis(var_file, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--name", "LOCOMUJOCO_CUSTOM_MODELS_PATH",
                                      "--value", "/m1", "--type", "list"])
    ds.add_variable_cli()
    assert _load(var_file)["LOCOMUJOCO_CUSTOM_MODELS_PATH"] == ["/m1"]

    monkeypatch.setattr(sys, "argv", ["prog", "--name", "LOCOMUJOCO_CUSTOM_MODELS_PATH",
                                      "--value", "/m1"])
    ds.remove_variable_cli()
    assert "LOCOMUJOCO_CUSTOM_MODELS_PATH" not in _load(var_file)


# --------------------------------------------------------------------------- #
# show_variable_cli
# --------------------------------------------------------------------------- #
def test_show_variable_cli_variants(var_file, monkeypatch, capsys):
    # no args at all -> parser.error -> SystemExit
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        ds.show_variable_cli()

    # --all on a non-existent file
    monkeypatch.setattr(sys, "argv", ["prog", "--all"])
    ds.show_variable_cli()
    assert "No variables file" in capsys.readouterr().out

    # populate a str var and a list var
    ds.add_variable("LOCOMUJOCO_AMASS_PATH", "/a", quiet=True)
    ds.add_variable("LOCOMUJOCO_CUSTOM_MODELS_PATH", "/m1", quiet=True)

    # --all with content (covers list + str printing)
    monkeypatch.setattr(sys, "argv", ["prog", "--all"])
    ds.show_variable_cli()
    out = capsys.readouterr().out
    assert "LOCOMUJOCO_AMASS_PATH" in out and "LOCOMUJOCO_CUSTOM_MODELS_PATH" in out

    # --name of an existing variable
    monkeypatch.setattr(sys, "argv", ["prog", "--name", "LOCOMUJOCO_AMASS_PATH"])
    ds.show_variable_cli()
    assert "/a" in capsys.readouterr().out

    # --name of a missing variable
    monkeypatch.setattr(sys, "argv", ["prog", "--name", "NOT_SET"])
    ds.show_variable_cli()
    assert "is not set" in capsys.readouterr().out


def test_show_variable_cli_empty_file(var_file, monkeypatch, capsys):
    # an existing-but-empty variables file with --all
    with open(var_file, "w") as f:
        yaml.dump({}, f)
    monkeypatch.setattr(sys, "argv", ["prog", "--all"])
    ds.show_variable_cli()
    assert "(no variables set)" in capsys.readouterr().out


def test_add_to_list_migration_and_guards(var_file):
    # a str value under a list-typed key auto-migrates to a single-element list
    ds._set_path_in_yaml_conf("/old", "LOCOMUJOCO_CUSTOM_MODELS_PATH", str(var_file), quiet=True)
    ds._add_path_to_yaml_list("/new", "LOCOMUJOCO_CUSTOM_MODELS_PATH", str(var_file), quiet=False)
    assert _load(var_file)["LOCOMUJOCO_CUSTOM_MODELS_PATH"] == ["/old", "/new"]

    # duplicate insert with prints enabled (non-quiet "already exists" branch)
    ds._add_path_to_yaml_list("/new", "LOCOMUJOCO_CUSTOM_MODELS_PATH", str(var_file), quiet=False)
    assert _load(var_file)["LOCOMUJOCO_CUSTOM_MODELS_PATH"] == ["/old", "/new"]

    # a non-list, non-str value under the key is rejected by both list helpers
    with open(var_file, "w") as f:
        yaml.dump({"LOCOMUJOCO_CUSTOM_MODELS_PATH": 5}, f)
    with pytest.raises(ValueError):
        ds._add_path_to_yaml_list("/x", "LOCOMUJOCO_CUSTOM_MODELS_PATH", str(var_file))
    with pytest.raises(ValueError):
        ds._remove_path_from_yaml_list("/x", "LOCOMUJOCO_CUSTOM_MODELS_PATH", str(var_file))


def test_remove_variable_unsupported_type(var_file):
    # a stored value that is neither str nor list -> TypeError
    with open(var_file, "w") as f:
        yaml.dump({"WEIRD_VAR": 5}, f)
    with pytest.raises(TypeError):
        ds.remove_variable("WEIRD_VAR")
