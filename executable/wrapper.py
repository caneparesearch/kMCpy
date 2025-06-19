#!/usr/bin/env python
"""a Wrapper for running the KMC, receive 1 standard input of file path, read the file as the incar and run KMC

Usage:

./wrapper.py ../examples/test_input.json

Raises:
    NotImplementedError: _description_
"""


from kmcpy.io import InputSet
from kmcpy.kmc import KMC
from gooey import Gooey, GooeyParser

import argparse


@Gooey
def main(**kwargs):
    """
    This is the wrapper for executing KMC

    """
    parser = GooeyParser(description="My Cool GUI Program!")
    parser.add_argument("Filename", widget="FileChooser")
    parser.add_argument("Date", widget="DateChooser")
    # parser = argparse.ArgumentParser()
    parser.add_argument('Inputfile', metavar='N', type=str,help='path to the input.json')
    args = parser.parse_args()
    inputset = InputSet.from_json(args.Inputfile)

    # initialize global occupation and conditions
    kmc = KMC.from_inputset(inputset = inputset)

    # run kmc
    kmc.run(inputset = inputset)

if __name__ == "__main__":

    main()
