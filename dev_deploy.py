import os


def write_rst_for_sphinx(filename="pymatgen_cif.py",api_doc_path="doc/source/modules/",module="kmcpy.external",package="pymatgen_cif"):
    with open(api_doc_path+filename.replace(".py",".rst"),"w+") as rst:
        rst.write("""
package
============

.. automodule:: module.package
    :members:
    :inherited-members:
                  """.replace("package",package).replace("module",module))
    return package
    

modules=[]

for root, dirs, files in os.walk("./kmcpy", topdown=False):
    for name in files:
        filename=os.path.join(root, name)
        if filename[-3:]==".py":
            print(root,name," is python script")
            
            package=name.replace(".py","")
            module_name=root.replace("./","").replace("/",".")
            
            write_rst_for_sphinx(filename=name,api_doc_path="doc/source/modules",module=module_name,package=package)
