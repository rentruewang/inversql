# Copyright (c) InverSQL Authors - All Rights Reserved

OS := $(shell uname -s)
PYTEST_FLAGS := 
PYTHON_VERSION :=
CHECK :=
CHECK_FLAG := $(if $(CHECK),--check,)
SUDO := sudo -E

# Running in GitHub Actions
ifeq ($(GITHUB_ACTIONS),true)
    SH := python3 ci/group-actions.py bash
	CLEANUP := $(SUDO) $(SH) ci/cleanup-github.sh
# Running locally
else
    SH := bash
	CLEANUP :=
endif

cleanup:
	$(CLEANUP)

publish:
	$(SH) ci/pdm.sh publish

build:
	$(SH) ci/pdm.sh build

install:
	$(SH) ci/pdm.sh install "-G:all"

sync:
	$(SH) ci/pdm.sh sync "-G:all"

pytest:
	$(SH) ci/pdm.sh run pytest $(PYTEST_FLAGS)

autoflake:
	$(SH) ci/pdm.sh run autoflake . $(CHECK_FLAG)

black:
	$(SH) ci/pdm.sh run black . $(CHECK_FLAG)

isort:
	$(SH) ci/pdm.sh run isort . $(CHECK_FLAG)

mypy:
	$(SH) ci/pdm.sh run mypy --install-types --non-interactive src
