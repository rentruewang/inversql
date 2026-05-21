# Copyright (c) InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import functools
import os

import nox


@nox.session
def publish(session: nox.Session):
    pdm(session).publish()


@nox.session
def build(session: nox.Session):
    pdm(session).build()


@nox.session
def pre_commit(session: nox.Session):
    formatting(session)
    typing(session)


@nox.session
def testing(session: nox.Session):
    _ = session.run("pytest", *session.posargs)


@nox.session
def formatting(session: nox.Session):
    autoflake(session)
    isort(session)
    black(session)
    nb_clean(session)


@nox.session
def formatting_check(session: nox.Session):
    autoflake_check(session)
    isort_check(session)
    black_check(session)
    nb_check(session)


@nox.session
def autoflake(session: nox.Session):
    _cmd(session, "autoflake", False)


@nox.session
def autoflake_check(session: nox.Session):
    _cmd(session, "autoflake", True)


@nox.session
def isort(session: nox.Session):
    _cmd(session, "isort", False)


@nox.session
def isort_check(session: nox.Session):
    _cmd(session, "isort", True)


@nox.session
def black(session: nox.Session):
    _cmd(session, "black", False)


@nox.session
def black_check(session: nox.Session):
    _cmd(session, "black", True)


@nox.session
def nb_clean(session: nox.Session):
    "Call `nb-clean clean`."
    pdm(session).run("nb-clean", "clean", "notebooks")


@nox.session
def nb_check(session: nox.Session):
    "Call `nb-clean check`."
    pdm(session).run("nb-clean", "check", "notebooks")


def _cmd(session: nox.Session, command: str, check: bool):
    check_flag = ["--check"] if check else []
    _ = pdm(session).run(command, *check_flag, ".")


@nox.session
def mypy(session: nox.Session):
    _ = session.run_always("mypy", "--install-types", "--non-interactive", "src")


@nox.session
def typing(session: nox.Session):
    mypy(session)


@functools.cache
def github(session: nox.Session):
    return _Github(session)


@functools.cache
def pdm(session: nox.Session):
    "Global singleton of `pdm`."
    return _Pdm(session)


@dcls.dataclass(frozen=True)
class _Github:
    "The manager for setting up github."

    session: nox.Session
    "The nox session to use."

    @functools.cache
    def setup(self) -> None:
        "The shared entrypoint to GitHub Actions scripts"

        # Does nothing outside of GitHub Actions.
        if not self.active():
            return

        self._remove_unwanted_files()
        self._log_storage_usage()

    def _run(self, *args: str):
        self.session.run_install(*args, external=True)

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
class _Pdm:
    "The manager for running `pdm` commands."

    session: nox.Session

    def __post_init__(self):
        github(self.session).setup()

        if _is_remote(self.session):
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
        if not _is_remote(self.session):
            return

        self.session.run_install("pdm", mode, "-G:all")

    def _run(self, *args: str):
        self.session.run(*args, external=True)


def _is_remote(session: nox.Session):
    return github(session).active()
