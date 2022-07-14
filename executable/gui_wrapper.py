#!pythonw
"""
Example program to demonstrate Gooey's presentation of subparsers
"""
import os
import argparse
from secrets import choice

from gooey import Gooey, GooeyParser
from kmcpy.io import InputSet,load_occ
from kmcpy.kmc import KMC
from kmcpy.event_generator import generate_events3
from kmcpy.model import LocalClusterExpansion
from kmcpy.external.pymatgen_structure import Structure

@Gooey(optional_cols=2, program_name="kmcpy",default_size=(1024, 768))
def main():
    settings_msg = 'kmcpy'
    parser = GooeyParser(description=settings_msg)
    parser.add_argument('--verbose', help='be verbose', dest='verbose',
                        action='store_true', default=False)
    
    subs = parser.add_subparsers(help='commands', dest='command')
    
    
    
    # event generator + local cluster expansion
    event_parser = subs.add_parser(
        'event', help='generate files required for local cluster expansion')
    event_parser.add_argument('prim_cif_name',
                             help='path to the cif file for generating local cluster expansion',
                             type=str, widget='FileChooser',default="/Users/weihangxie/Documents/GitHub/kmcPy_dev/dev/v3_nasicon_bulk/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif")
    event_parser.add_argument("convert_to_primitive_cell",choices=["yes","no"],default="yes")
    event_parser.add_argument("mobile_ion_identifier_type",choices=["label","specie"],default="label")
    event_parser.add_argument("mobile_ion_specie_1_identifier",default="Na1")
    event_parser.add_argument("mobile_ion_specie_2_identifier",default="Na2")
    event_parser.add_argument("local_env_cutoff_dict",default="Na+,Na+,4.0;Na+,Si4+,4.0")    
    event_parser.add_argument("cutoff_region",default=4.0)
    event_parser.add_argument("cutoff_for_point_cluster",default=10,type=int)
    event_parser.add_argument("cutoff_for_pair_cluster",default=6,type=int)
    event_parser.add_argument("cutoff_for_triplet_cluster",default=6,type=int)
    event_parser.add_argument("cutoff_for_quadruplet_cluster",default=0,type=int)
    event_parser.add_argument("species_to_be_removed",default="Zr4+,O2-,O,Zr")
    event_parser.add_argument("events_output_dir",default="/Users/weihangxie/Documents/GitHub/kmcPy_dev/dev/v3_nasicon_bulk/input",widget="DirChooser")
    event_parser.add_argument("supercell_shape",default="2,1,1")
    #event_parser.add_argument("generate_events")
    
    # output
    event_parser.add_argument("verbosity",choices=["INFO","WARNING","CRITICAL"],default="WARNING")
    event_parser.add_argument("--is_write_basis",action="store_true")
    event_parser.add_argument("--export_local_env_structure",action="store_true")


    # things that shouldn't change at all
    event_parser.add_argument("event_fname",default="events.json")
    event_parser.add_argument("event_kernal_fname",default="event_kernal.csv")
    event_parser.add_argument("distance_matrix_rtol",default=0.01,type=float)
    event_parser.add_argument("distance_matrix_atol",default=0.01,type=float)
    event_parser.add_argument("find_nearest_if_fail",default=False)
    event_parser.add_argument("local_cluster_expansion_json",default="lce.json")
    


    # kmc task
    kmc_parser = subs.add_parser(
        'kmc', help='generate files required for local cluster expansion')
    kmc_parser.add_argument('CifPath',
                             help='path to the cif file for generating local cluster expansion',
                             type=str, widget='FileChooser',default="/Users/weihangxie/Documents/GitHub/kmcPy_dev/dev/v3_nasicon_bulk/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif")    


    args=parser.parse_args()

    #print(vars(args))
    
    if args.command=="event":
        
        # species to be removed
        args.species_to_be_removed=args.species_to_be_removed.split(",")
        
        # convert to primitive cell
        if args.convert_to_primitive_cell=="yes":
            args.convert_to_primitive_cell=True
        else:
            args.convert_to_primitive_cell=False
            
        # cutoff cluster
        cutoff_cluster=[int(args.cutoff_for_pair_cluster),int(args.cutoff_for_triplet_cluster),int(args.cutoff_for_quadruplet_cluster)]
        
        
        # local env cutoff dict
        tmp_local_env_cutoff_dict={}
        
        for cutoff in args.local_env_cutoff_dict.split(";"):
            cutoff=cutoff.split(",")
            tmp_local_env_cutoff_dict[(str(cutoff[0]),str(cutoff[1]))]=float(cutoff[2])
        
        args.local_env_cutoff_dict=tmp_local_env_cutoff_dict
        
        # supercell
        args.supercell_shape=[int(scale) for scale in args.supercell_shape.split(",")]
        
        #fanme_path
        args.event_fname=os.path.join(args.events_output_dir,args.event_fname)
        args.event_kernal_fname=os.path.join(args.events_output_dir,args.event_kernal_fname)
        
        print((vars(args)))
        
        a=LocalClusterExpansion(api=3)
        a.initialization3(cutoff_cluster=cutoff_cluster,template_cif_fname=args.prim_cif_name,**vars(args))
        a.to_json(args.events_output_dir+"/"+args.local_cluster_expansion_json)
        
        generate_events3(**vars(args))
        
        
        



if __name__ == '__main__':
    main()