# Contributing

## Development installation

First, clone and navigate into the project:

```bash
git clone https://github.com/rentruewang/inversql
cd inversql/
```

Alternatively, use ssh:
```bash
git clone git@github.com:rentruewang/inversql
cd inversql/
```

I'm using [pdm](https://pdm-project.org/) in this project for dependency management.
To install all dependencies (including development dependencies) with `pdm`, run

```bash
pdm install -G:all
```

Alternatively, use of `pip` is also allowed (although might be less robust due to lack of version solving), and the dependencies are frozen in `requirements.txt` (managed by `pdm` and `nox` commands).

```bash
pip install -r requirements.txt
```

Both commands result in an editable installation.

I'm also using [nox](https://nox.thea.codes/en/stable/) for running miscellaneous commands.
They are quite comprehensive and powers our github action scripts.
Check there for some examples if you're not familiar with `nox`.

## Recommended development style

### Python code style

Please write code matching the style of the surrounding code.

Otherwise, follow the following style guide that I personally use (by me): [link](https://github.com/rentruewang/mind/blob/main/CONTRIBUTING.md).

### Documentation

The documentation string style follows the Google style format specified [here](https://mkdocstrings.github.io/griffe/docstrings/#google-style).

### Formatting

Use `autoflake`, `isort`, `black` for consistent formatting.

Prior to commiting, please run the following commands:

```bash
autoflake .
isort .
black .
```

### Commit message

Commit message should follow the format:

```
{emoji} Message. [([fix] #{issue})]

Detailed explanation.
```

Where `[]` denotes optional in the above commit message. `{emoji}` should be a relevant emoji, and `#{issue}` should be the relevant issue number.

### Typing

Be sure to run `mypy` prior to submitting! There can be issue with `mypy` not finding libraries. The command I use for checking is

```bash
mypy .
```
