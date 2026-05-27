# Reference Overview

Use the reference section when you need exact API names, class signatures, or
module-level documentation. For task-oriented workflows, start with the
[workflow tutorial](../tutorial/index.md) or
[advanced pages](../howto/index.md).

## API Reference

The API reference is generated from the Python source and docstrings:

- [API reference](../modules/api.rst)

## Configuration Fields

To inspect valid configuration fields from your installed version:

```shell
uv run python -c "from kmcpy.simulator.config import Configuration; Configuration.help_fields()"
```

This is often the fastest way to check whether a field belongs to the physical
system configuration or runtime controls.

```{toctree}
:maxdepth: 1
:hidden:

../modules/api
```
