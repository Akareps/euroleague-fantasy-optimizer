"""Data adapters.

Every adapter turns some external source into the plain dataclasses in
:mod:`elfantasy.models`. Nothing downstream knows or cares where the numbers
came from, which is what lets the project run fully offline against fixtures.

Public feeds change without notice. If a fetch starts returning nothing, the
adapter -- not the model -- is what needs fixing, and
``elfantasy sync --debug`` prints the raw payload that confused it.
"""

from elfantasy.data.http import HttpClient  # noqa: F401
