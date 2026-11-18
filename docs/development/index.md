# Development

This section is for people who work on LocoMuJoCo itself: how the code is organised, how it is tested,
how the documentation is written and how a release is made. It also holds the project pages: the changelog,
a comparison to related software, and how to cite the library.

LocoMuJoCo is developed in the open at [github.com/robfiras/loco-mujoco](https://github.com/robfiras/loco-mujoco)
under the MIT license. Bugs and questions go to the
[issue tracker](https://github.com/robfiras/loco-mujoco/issues); pull requests target the `dev` branch.

## Contributing

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

Development setup, the branch model, and what a pull request should contain.
:::

:::{grid-item-card} Architecture for developers
:link: architecture
:link-type: doc

Package layout, the class hierarchy, the order of calls in `reset` and `step`, and the design rules.
:::

:::{grid-item-card} Testing
:link: testing
:link-type: doc

Running the test suite, what it covers, coverage measurement and continuous integration.
:::

:::{grid-item-card} Writing documentation
:link: documentation
:link-type: doc

Building the documentation locally and the conventions for pages and docstrings.
:::

:::{grid-item-card} Release process
:link: releasing
:link-type: doc

Versioning, branches, tags, packaging, and the model and dataset repositories.
:::
::::

## Project

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Changelog
:link: changelog
:link-type: doc

What changed in each release, including breaking changes and how to migrate.
:::

:::{grid-item-card} Comparison to related software
:link: comparison
:link-type: doc

How LocoMuJoCo relates to other humanoid learning libraries and retargeting pipelines.
:::

:::{grid-item-card} Citing LocoMuJoCo
:link: citation
:link-type: doc

The reference for the library, the datasets and models to cite, and the licenses.
:::
::::

```{toctree}
:hidden:
:caption: Contributing
:maxdepth: 1

contributing
architecture
testing
documentation
releasing
```

```{toctree}
:hidden:
:caption: Project
:maxdepth: 1

changelog
comparison
citation
```
