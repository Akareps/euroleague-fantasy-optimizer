"""Solver selection, PuLP 3/4 compatibility and a couple of platform work-arounds.

CBC comes with the install (bundled in PuLP 3, the ``pulp[cbc]`` extra in
PuLP 4), so there is nothing else to set up. PuLP 4 moved variable creation
onto the problem and returns solve statistics instead of setting a status;
:func:`binary` and :func:`solve` hide both differences. Two things still go
wrong in practice and both produce baffling errors, so they are handled here
once:

* **Windows path limits.** CBC is invoked as a subprocess, and both its own
  executable path and the temporary model files it writes count against the
  260-character ``MAX_PATH`` limit. Installing into a deeply nested directory
  (an OneDrive-synced ``AppData`` tree, say) yields
  ``FileNotFoundError: [WinError 206] The filename or extension is too long``,
  which says nothing about the real cause. We point CBC's scratch files at the
  system temp directory and translate the error into an actionable message.
* **Solver time limits.** The default is unbounded. A pathological instance can
  hang; ``ELFANTASY_SOLVER_TIMEOUT`` caps it.
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings

import pulp

log = logging.getLogger(__name__)


class SolverUnavailableError(RuntimeError):
    """CBC could not be run at all -- an environment problem, not a modelling one."""


def default_solver(*, time_limit: float | None = None, msg: bool = False) -> pulp.LpSolver:
    """Return a configured CBC solver.

    PuLP 3's ``PULP_CBC_CMD`` bundles a CBC binary. PuLP 4 removed it; there
    ``COIN_CMD`` finds the binary that the ``pulp[cbc]`` extra installs. The
    deprecation warning is suppressed because there is nothing a user of this
    project can act on.
    """

    limit = time_limit
    if limit is None:
        env = os.getenv("ELFANTASY_SOLVER_TIMEOUT")
        limit = float(env) if env else None

    kwargs: dict = {"msg": msg}
    if limit:
        kwargs["timeLimit"] = limit

    factory = getattr(pulp, "PULP_CBC_CMD", None) or getattr(pulp, "COIN_CMD", None)
    if factory is None:  # pragma: no cover - no CBC binding at all
        raise SolverUnavailableError(
            "PuLP exposes neither PULP_CBC_CMD nor COIN_CMD. Install a CBC "
            "binding with `pip install 'pulp[cbc]'`."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        solver = factory(**kwargs)

    # Keep CBC's scratch files short and out of the project tree.
    solver.tmpDir = tempfile.gettempdir()
    return solver


def binary(problem: pulp.LpProblem, name: str) -> pulp.LpVariable:
    """A 0/1 variable belonging to ``problem`` (PuLP 4 creates variables on the
    problem; older PuLP has no ``add_variable``)."""

    add = getattr(problem, "add_variable", None)
    if add is not None:
        return add(name, cat="Binary")
    return pulp.LpVariable(name, cat="Binary")  # pragma: no cover - PuLP < 3.3


# PuLP 4 status names -> the PuLP 3 strings the optimisers check. PuLP 3 reports
# a CBC run stopped by a limit with a solution in hand as "Optimal"; keep that.
_LIMITS = {"TimeLimit", "NodeLimit", "GapLimit", "IterationLimit", "SolutionLimit", "Stopped"}


def _status_name(result, problem: pulp.LpProblem) -> str:
    stats = getattr(result, "status", None)
    if stats is None:  # PuLP 3: solve() returns an int and sets problem.status
        return pulp.LpStatus[problem.status]
    name = getattr(stats, "name", str(stats))
    if name in _LIMITS and getattr(result, "has_solution", False):
        return "Optimal"
    return "Not Solved" if name == "NotSolved" else name


def solve(problem: pulp.LpProblem, solver: pulp.LpSolver | None = None) -> str:
    """Solve, converting environment failures into a message that helps.

    Returns the PuLP status string; raises :class:`SolverUnavailableError` if
    CBC itself could not be executed.
    """

    solver = solver or default_solver()
    try:
        result = problem.solve(solver)
    except (FileNotFoundError, OSError) as exc:
        winerror = getattr(exc, "winerror", None)
        if winerror == 206 or "too long" in str(exc).lower():
            raise SolverUnavailableError(
                "CBC could not start because the path to it is too long for Windows "
                "(MAX_PATH = 260 characters).\n"
                f"  solver path: {getattr(solver, 'path', 'unknown')}\n"
                "Fix by doing one of:\n"
                "  - reinstall this project somewhere shallow, e.g. C:\\dev\\elfantasy\n"
                "  - create the virtualenv at a short path (py -m venv C:\\venvs\\elf)\n"
                "  - enable long paths: in an elevated PowerShell, "
                "Set-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' "
                "LongPathsEnabled 1, then reboot"
            ) from exc
        raise SolverUnavailableError(
            f"could not run the CBC solver: {exc}. Install a solver PuLP can find "
            f"(e.g. `pip install pulp --force-reinstall`) or pass your own via the "
            f"`solver=` argument."
        ) from exc
    return _status_name(result, problem)
