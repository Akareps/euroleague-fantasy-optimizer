"""Where the default config and data live, in a checkout and when installed."""

from __future__ import annotations

from elfantasy import config


def test_a_source_checkout_uses_the_repository_folders():
    assert config.default_config_dir() == config.REPO_ROOT / "config"
    assert config.default_data_dir() == config.REPO_ROOT / "data"


def test_an_installed_package_falls_back_to_its_packaged_config(tmp_path):
    # Installed, there is no config/ two levels above the package.
    assert config.default_config_dir(tmp_path) == config.PACKAGED_CONFIG_DIR
    assert config.default_data_dir(tmp_path) != tmp_path / "data"


def test_the_wheel_ships_the_config():
    pyproject = (config.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"config" = "elfantasy/_config"' in pyproject
