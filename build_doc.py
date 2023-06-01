import os

api_doc_path="docs/source/modules/"

def write_rst_for_sphinx(filename="pymatgen_cif.py",api_doc_path="docs/source/modules/",module="kmcpy.external",package="pymatgen_cif"):
    with open(api_doc_path+filename.replace(".py",".rst"),"w+") as rst:
        rst.write("""package
=========================

.. automodule:: modulename.package
    :members:
    :inherited-members:
                  """.replace("package",package).replace("modulename",module))
    return package
    

api_list=[]

for root, dirs, files in os.walk("./kmcpy", topdown=False):
    for name in files:
        filename=os.path.join(root, name)
        if filename[-3:]==".py" and ("__init__" not in filename) and ("_version" not in filename) and ("tools" not in root) and ("external" not in root) and ("sites_and_lattices" not in filename):
            # for now, skip the tools 
            # need to modify documentation
            
            print(root,name," is python script")
            
            package=name.replace(".py","")
            module_name=root.replace("./","").replace("/",".")
            
            write_rst_for_sphinx(filename=name,api_doc_path=api_doc_path,module=module_name,package=package)
            api_list.append(package)

with open(api_doc_path+"api.rst","w+") as api_file:
    filestring="""API Reference Documentation
===========================

.. toctree::
    :maxdepth: 2
    :caption: Contents:

"""
    for api in api_list:
        filestring+="    "+api+".rst\n"
    api_file.write(filestring)
    
os.chdir("docs")
os.system("make html")
