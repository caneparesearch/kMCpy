from kmcpy.external.pymatgen_structure import Structure
a=Structure.from_cif("/Users/weihangxie/Documents/GitHub/kmcPy_dev/dev/NVP_test/refill-N4VP-POS.cif",primitive=True)
a.to('cif',"prim.cif",symprec=0.01)