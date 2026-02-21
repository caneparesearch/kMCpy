import os

api_doc_path = "docs/source/modules/"
excluded_packages = {"gui_wrapper"}


def write_rst_for_sphinx(
    filename="pymatgen_cif.py",
    api_doc_path="docs/source/modules/",
    module="kmcpy.external",
    package="pymatgen_cif",
):
    header = """package
=========================

""".replace("package", package)

    if package == "config":
        header += """Parameter discovery
-------------------

Use ``SimulationConfig.help_parameters()`` to list valid parameter names and see how
parameters are split between ``system_config`` and ``runtime_config``.

.. code-block:: python

    from kmcpy.simulator.config import SimulationConfig

    SimulationConfig.help_parameters()

"""

    with open(api_doc_path + filename.replace(".py", ".rst"), "w+") as rst:
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


api_list = []

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

            write_rst_for_sphinx(
                filename=name,
                api_doc_path=api_doc_path,
                module=module_name,
                package=package,
            )
            api_list.append(package)

with open(api_doc_path + "api.rst", "w+") as api_file:
    filestring = """API Reference Documentation
===========================

.. toctree::
    :maxdepth: 4
    :caption: Contents:

"""
    # Keep first-seen order while removing duplicates from same package names.
    unique_api_list = list(dict.fromkeys(api_list))
    for api in unique_api_list:
        filestring += "    " + api + ".rst\n"
    api_file.write(filestring)

os.chdir("docs")
os.system("make html")
