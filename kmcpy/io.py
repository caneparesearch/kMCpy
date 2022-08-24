"""
IO takes dictionary like object and convert them into json writable string

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import numpy as np
import json

# class IO:
#     def __init__(self):
#         pass

#     def to_json(self,fname):
#         print('Saving:',fname)
#         with open(fname,'w') as fhandle:
#             d = self.as_dict()
#             jsonStr = json.dumps(d,indent=4,default=convert) # to get rid of errors of int64
#             fhandle.write(jsonStr)
    
def convert(o):
    if isinstance(o, np.int64): return int(o)
    elif isinstance(o, np.int32): return int(o)  
    raise TypeError


def load_occ(fname="./initial_state.json",shape=[2,1,1],select_sites=[0,1,2,3,4,5,6,7,12,13,14,15,16,17],api=1,verbose=False,**kwargs):
    """load occupation data

    Args:
        fname (str, optional): initial occupation that also includes immutable site(for example, Zr, O). Defaults to "./initial_state.json".
        shape (list, optional): supercell shape. Defaults to [2,1,1].
        select_sites (list, optional): all the sites included in kinetic monte carlo process, i.e., this is the list include only the indices of Na, Si, P (no Zr and O) in the Na1+xZr2P3-xSixO12.  . Defaults to [0,1,2,3,4,5,6,7,12,13,14,15,16,17].
        api (int, optional): version. Defaults to 1.
        verbose (bool, optional): verbose output. Defaults to False.

    Raises:
        ValueError: 

    Returns:
        chebyshev occupation: list of 1 and -1 states, the initial occupation data of sites included in KMC, for example, Na, Si, P initial states in NZSP
    """
    if api==1:
        with open(fname,'r') as f:
            occupation = (np.array(json.load(f)['occupation']).reshape((42,)+(shape[0],shape[1],shape[2]))[select_sites].flatten('C')) # the global occupation array in the format of (site,x,y,z)
        occupation_chebyshev = np.where(occupation==0, -1, occupation)  # replace 0 with -1 for Chebyshev basis
        return occupation_chebyshev
    if api==2 or api==3:
        
        with open(fname,'r') as f:
            
            # read the occupation from json
            occupation_raw_data=np.array(json.load(f)['occupation'])
            
            # check if the occupation is compatible with the shape.
            # for example. if there is 20 occupation data and supercell is [3,1,1], it is incompatible because 20/3 is not integer
            if len(occupation_raw_data)%(shape[0]*shape[1]*shape[2])!=0:
                raise ValueError("change log: The dimension conversion of occupation is now flexible: in API=1, the dimension is determined in the function code. In API=2, the dimension is automatically calculated from supercell. Therefore, You need to check if the length of occupation data is compatible with the supercell shape. The length of occupation data"+str(len(occupation_raw_data))+"is incompatible with the supercell shape,please check the "+fname)

            # this is the total sites, or global occupation array[0] for api=1
            site_nums=int(len(occupation_raw_data)/(shape[0]*shape[1]*shape[2]))
            
            # this is the dimension of global occupation array
            convert_to_dimension=(site_nums,shape[0],shape[1],shape[2])
            
            occupation = (occupation_raw_data.reshape(convert_to_dimension)[select_sites].flatten('C')) # the global occupation array in the format of (site,x,y,z). Now it only contain the selected sites.
            
            occupation_chebyshev = np.where(occupation==0, -1, occupation)  # replace 0 with -1 for Chebyshev basis
            
            if verbose:
                print("verbose output is enabled, this is function io.load_acc from Api2")
                print("selected sites are",select_sites)
                print("converting the occupation raw data to dimension:",(convert_to_dimension))
                print("occupation_chebyshev:",occupation_chebyshev)
            
        return occupation_chebyshev
    

# to be developed

class InputSet:
    """
    a flexible input set class for running KMC
    just a dictionary
    """
    def __init__(self,_parameters={},api=1) -> None:
        
        self._parameters=_parameters
        self.api=api
        self._parameters["api"]=api
        pass
    
    @classmethod
    def from_json(self,input_json_path=r"examples/test_input.json",verbose=False,api=1):
        """
        input_reader takes input (a json file with all parameters as shown in run_kmc.py in examples folder)
        return a dictionary with all input parameters
        if verbose=True, then print the parameters
        """
        _parameters=json.load(open(input_json_path))
        if verbose:
            self.report_parameter()
        return InputSet(_parameters,api)

    def report_parameter(self,format="equation"):
        """
        report_parameter, to print the parameters of this input set. This is only for development convenience
        
        for example: the output of a default input set is :v= 5000000000000,equ_pass= 1,kmc_pass= 1000,supercell_shape= [2, 1, 1],fitting_results='./inputs/fitting_results.json',fitting_results_site='./inputs/fitting_results_site.json',lce_fname='./inputs/lce.json',lce_site_fname='./inputs/lce_site.json',prim_fname='./inputs/prim.json',event_fname='./inputs/events.json',event_kernel='./inputs/event_kernal.csv',mc_results='./initial_state.json',T= 298,comp= 1,structure_idx= 1,occ= [-1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1 -1  1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1].
        Args:
            format (str, optional): "equation" or "dict". If format=dict, then print a python dict. format=equation: print equations that is capable for **kwargs.  Defaults to "equation".
        """
        if format=="dict":
            print(self._parameters)
        if format=="equation":
            for i in self._parameters:
                print(i,type(self._parameters[i]))
            for i in self._parameters:
                if type(self._parameters[i]) is str:
                    print(str(i)+"='"+self._parameters[i]+"'",end=",")
                elif type(self._parameters[i]) is np.ndarray:
                    print(str(i)+"=",self._parameters[i].tolist(),end=",")                  
                else:
                    print(str(i)+"=",self._parameters[i],end=",")
    
    def set_parameter(self,key_to_change="T",value_to_change=273.15):
        """_summary_

        Args:
            key_to_change (str, optional): the key to change in the parameters. Defaults to "T".
            value_to_change (any, optional): any type that json can read. Defaults to 273.15.
        """
        self._parameters[key_to_change]=value_to_change
    
    def enumerate(self,**kwargs):
        """generate a new InputSet from the input kwargs

        Inputs:
            for example: InputSet.enumerate(T=298.15)
        
        Returns:
            InputSet: a InputSet class with modified parameters
        """
         
        new_InputSet=InputSet(self._parameters)
        for key_to_change in kwargs:
            new_InputSet.set_parameter(key_to_change,kwargs[key_to_change])
        return new_InputSet
    
    def change_key_name(self,oldname="lce",newname="lce_fname"):
        """change the key name from old name to new name for self._parameters

        Args:
            oldname (str): defined name in self._parameters
            newname (str): new name
        """
        self._parameters[newname]=self._parameters[oldname]
    
    def api_converter(self):
        if self.api==1:
            pass
        if self.api>1:
            raise NotImplementedError
        pass
    
    def parameter_checker(self):
        """a rough parameter checker to make sure that there is enough parameters to run a job

        Raises:
            ValueError: In case that parameter is not defined in the self._parameters
        """
        
        
        if self.api==1:
            for i in ['v', 'equ_pass', 'kmc_pass', 'supercell_shape', 'fitting_results', 'fitting_results_site', 'lce_fname', 'lce_site_fname', 'prim_fname', 'event_fname', 'event_kernel', 'mc_results', 'T', 'comp', 'structure_idx']:
                if i not in self._parameters:
                    print(i+" is not defined yet in the parameters!")
                    raise ValueError('This program is exploding due to undefined parameter.')
        elif self.api==2:
            for i in ['v', 'equ_pass', 'kmc_pass', 'supercell_shape', 'fitting_results', 'fitting_results_site', 'lce_fname', 'lce_site_fname', 'prim_fname', 'event_fname', 'event_kernel', 'mc_results', 'T', 'comp', 'structure_idx',"select_sites"]:
                if i not in self._parameters:
                    print(i+" is not defined yet in the parameters!")
                    raise ValueError('This program is exploding due to undefined parameter.')
        elif self.api==3:
            for i in ['v', 'equ_pass', 'kmc_pass', 'supercell_shape', 'fitting_results', 'fitting_results_site', 'lce_fname', 'lce_site_fname', 'prim_fname', 'event_fname', 'event_kernel', 'mc_results', 'T', 'comp','dimension','q',"select_sites"]:
                if i not in self._parameters:
                    print(i+" is not defined yet in the parameters!")
                    raise ValueError('This program is exploding due to undefined parameter. Please check input json file')
        else:
            raise NotImplementedError("api version not implemented")