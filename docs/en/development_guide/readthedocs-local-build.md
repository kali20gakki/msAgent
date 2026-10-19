# Validating ReadTheDocs Builds Locally

This document describes how to validate the ReadTheDocs documentation build locally and is **not published to the ReadTheDocs website**.

## Setting Up the Environment

Ensure that Python 3.8+ and pip are installed.

## Installing Dependencies

```bash
cd <project-root-directory>
pip install -r docs/requirements.txt
```

> For example, if the project is located at `D:\code\msagent`, run `cd D:\code\msagent`.

## Building Locally

```bash
# Build the HTML documentation
sphinx-build -b html docs/ docs/_build/html/

# Or use the make command (Linux/Mac)
make -C docs html
```

## Viewing the Results

After the build completes, open the following file in a browser:

```text
<project-root-directory>/docs/_build/html/index.html
```

Alternatively, run the following command in a terminal (the `http` module of Python 3 must be installed):

```bash
# Change to the build directory
cd docs/_build/html/

# Python 3
python -m http.server 8000

# Then access http://localhost:8000/
```

> For example, if the project is located at `D:\code\msagent`, open `D:\code\msagent\docs\_build\html\index.html`.

## Frequently Asked Questions

### 1. Missing Dependencies

If you encounter a `ModuleNotFoundError`, check `docs/requirements.txt` and install all dependencies:

```bash
pip install -r docs/requirements.txt
```

### 2. Build Warnings

Sphinx may output some warnings. These warnings usually do not affect the build result. If you see an error (ERROR), fix it.

### 3. Chinese Character Display Issues

Ensure that `language = 'zh_CN'` is set in `docs/conf.py` and that the system supports Chinese fonts.

## Cleaning Up Build Files

```bash
# Windows
rmdir /s /q docs\_build

# Linux/Mac
rm -rf docs/_build
```

## Pushing to ReadTheDocs

After local validation passes, push the code to the remote repository. ReadTheDocs automatically detects code updates and rebuilds the documentation.

## Reference Links

- [ReadTheDocs configuration documentation](https://docs.readthedocs.io/en/stable/config-file/v2.html)
- [Sphinx documentation](https://www.sphinx-doc.org/)
