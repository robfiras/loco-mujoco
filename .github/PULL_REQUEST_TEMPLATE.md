<!-- Pull requests target the `dev` branch. See CONTRIBUTING.md for details. -->

## What does this change?

<!-- A short description, and the issue it closes if there is one (e.g. "Closes #123"). -->

## Checklist

- [ ] The pull request targets `dev`.
- [ ] Tests are added or updated, and `pytest --ignore=tests/test_task_factories.py` passes
      (or the affected test files, if the whole suite does not fit into memory).
- [ ] Environment components work on both backends (NumPy and JAX), or the description says why not.
- [ ] Public classes and functions have Google-style docstrings.
- [ ] The documentation is updated where the change is visible to users.
- [ ] An entry is added to the "Unreleased" section of `CHANGELOG.md`; breaking changes are marked.
