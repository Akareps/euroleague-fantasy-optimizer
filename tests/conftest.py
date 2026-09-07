from __future__ import annotations

import pytest

from elfantasy.config import load_settings
from elfantasy.data.sample import build_sample_dataset
from elfantasy.pipeline import Pipeline


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def dataset():
    return build_sample_dataset(seed=11)


@pytest.fixture(scope="session")
def pipeline(settings, dataset):
    return Pipeline.build(settings=settings, dataset=dataset)


@pytest.fixture(scope="session")
def model(settings):
    return settings.model
