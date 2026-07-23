# Copyright (c) InverSQL Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import functools
import os
from collections import abc as cabc

import nox

_session: nox.Session | None = None
"The global shared session."


def get_session() -> nox.Session:
    "Get the session set by `set_session`."

    assert _session
    return _session


@ctxl.contextmanager
def set_session(sess: nox.Session):
    "Set the session used by `get_session`."

    global _session
    before, _session = _session, sess

    try:
        yield sess
    finally:
        _session = before


def nox_session(function: cabc.Callable[[], None]) -> cabc.Callable[[], None]:

    @functools.wraps(function)
    def wrapper(session: nox.Session):
        with set_session(session):
            return function()

    _ = nox.session(wrapper)

    return function


@nox_session
def publish():
    Pdm().publish()


@nox_session
def build():
    Pdm().build()


@nox_session
def pre_commit():
    formatting()
    typing()


@nox_session
def testing():
    _ = Pdm().run("pytest", *get_session().posargs)


@nox_session
def formatting():
    autoflake()
    isort()
    black()
    lock()


@nox_session
def formatting_check():
    autoflake_check()
    isort_check()
    black_check()


@nox_session
def autoflake():
    _cmd("autoflake", False)


@nox_session
def autoflake_check():
    _cmd("autoflake", True)


@nox_session
def isort():
    _cmd("isort", False)


@nox_session
def isort_check():
    _cmd("isort", True)


@nox_session
def black():
    _cmd("black", False)


@nox_session
def black_check():
    _cmd("black", True)


def _cmd(command: str, check: bool):
    check_flag = ["--check"] if check else []
    _ = Pdm().run(command, *check_flag, ".")


@nox_session
def mypy():
    _ = Pdm().run("mypy", "--install-types", "--non-interactive", "src")


@nox_session
def lock():
    get_session().run(
        "pdm",
        "lock",
        "--refresh",
        "--strategy",
        "inherit_metadata",
    )

    get_session().run(
        "pdm",
        "export",
        "-o",
        "requirements.txt",
        "--without-hashes",
        "--self",
        external=True,
    )


@nox_session
def typing():
    mypy()


@dcls.dataclass(frozen=True)
class Github:
    "The manager for setting up github."

    @functools.cache
    def setup(self) -> None:
        "The shared entrypoint to GitHub Actions scripts"

        # Does nothing outside of GitHub Actions.
        if not self.active():
            return

        self._remove_unwanted_files()
        self._log_storage_usage()

    def _run(self, *args: str):
        get_session().run_install(*args, external=True)

    def _remove_unwanted_files(self) -> None:
        "Remove the files GitHub Actions pre-installed."

        print("Removing files we did not ask for...")

        for folder in [
            "/usr/local/lib/android",
            "/usr/share/dotnet",
            "/usr/local/.ghcup",
        ]:
            self._run("sudo", "rm", "-rf", folder)

        self._run("docker", "system", "prune", "-af", "--volumes")

    def _log_storage_usage(self) -> None:
        "Log how much usage is currently being used by GitHub Actions."
        print("Investigating how much storage is used in GitHub Actions...")

        self._run("df", "-h")

    @staticmethod
    def active() -> bool:
        "Detect whether or not it is running in GitHub Actions."

        print("Checking if we are in GitHub Actions...", end=" ")
        result = os.getenv("GITHUB_ACTIONS") == "true"
        print("Yes" if result else "No")
        return result


@dcls.dataclass(frozen=True)
class Pdm:
    "The manager for running `pdm` commands."

    def __post_init__(self):
        Github().setup()

        if _is_remote():
            self._run("pdm", "config", "python.use_venv", "true")

    def sync(self) -> None:
        self._sync_or_install("sync")

    def install(self):
        self._sync_or_install("install")

    def build(self):
        self.install()
        self._run("pdm", "build")

    def publish(self):
        self.install()
        self._run("pdm", "publish")

    def run(self, *args: str):
        self.sync()
        self._run("pdm", "run", *args)

    def _sync_or_install(self, mode: str) -> None:
        # Don't repeatedly reinstall locally.
        if not _is_remote():
            return

        get_session().run_install("pdm", mode, "-G:all")

    def _run(self, *args: str):
        get_session().run(*args, external=True)


def _is_remote():
    return Github().active()
