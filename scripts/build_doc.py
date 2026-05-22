import os
import subprocess
import sys

api_doc_path = "docs/source/modules/"
# `api.rst` is the generated toctree index page; exclude module `api.py` to
# avoid self-referential toctree entry (`api.rst` including `api.rst`).
excluded_packages = {"gui_wrapper", "api"}


def write_rst_for_sphinx(
    rst_filename="pymatgen_cif.rst",
    api_doc_path="docs/source/modules/",
    module="kmcpy.external",
    package="pymatgen_cif",
    title=None,
):
    title = title or package
    header = f"{title}\n{'=' * len(title)}\n\n"

    if package == "config":
        header += """Parameter discovery
-------------------

Use ``Configuration.help_parameters()`` to list valid parameter names and see how
parameters are split between ``system_config`` and ``runtime_config``.

.. code-block:: python

    from kmcpy.simulator.config import Configuration

    Configuration.help_parameters()

"""

    with open(api_doc_path + rst_filename, "w+") as rst:
        rst.write(header)
        rst.write(
            """.. automodule:: modulename.package
    :members:
    :inherited-members:
                  """.replace(
                "modulename", module
            )
            .replace("package", package)
        )
    return package


def rst_stem_for_module(module_name, package, duplicate_packages):
    if package not in duplicate_packages:
        return package
    module_suffix = module_name.replace("kmcpy.", "").replace(".", "_")
    return f"{module_suffix}_{package}"


module_entries = []

for root, dirs, files in os.walk("./kmcpy", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        if (
            filename[-3:] == ".py"
            and ("__init__" not in filename)
            and ("_version" not in filename)
            and ("tools" not in root)
            and ("external" not in root)
            and ("sites_and_lattices" not in filename)
        ):
            # for now, skip the tools
            # need to modify documentation

            print(root, name, " is python script")

            package = name.replace(".py", "")
            if package in excluded_packages:
                continue

            module_name = root.replace("./", "").replace("/", ".")
            module_entries.append((module_name, package))

package_counts = {}
for _, package in module_entries:
    package_counts[package] = package_counts.get(package, 0) + 1

duplicate_packages = {
    package for package, count in package_counts.items() if count > 1
}

api_list = []

for module_name, package in module_entries:
    rst_stem = rst_stem_for_module(
        module_name,
        package,
        duplicate_packages,
    )

    write_rst_for_sphinx(
        rst_filename=f"{rst_stem}.rst",
        api_doc_path=api_doc_path,
        module=module_name,
        package=package,
        title=rst_stem,
    )
    api_list.append(rst_stem)

with open(api_doc_path + "api.rst", "w+") as api_file:
    filestring = """API Reference Documentation
===========================

.. toctree::
    :maxdepth: 4
    :caption: Contents:

"""
    # Keep first-seen order while removing any exact duplicate output pages.
    unique_api_list = list(dict.fromkeys(api_list))
    for api in unique_api_list:
        filestring += "    " + api + ".rst\n"
    api_file.write(filestring)

subprocess.run(
    [sys.executable, "-m", "sphinx", "-b", "html", "source", "build/html"],
    cwd="docs",
    check=True,
)
