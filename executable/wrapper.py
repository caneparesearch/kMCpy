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
    # parser.add_argument('incar', metavar='N', type=str,help='path to the input.json')
    args = parser.parse_args()
    inputset = InputSet.from_json(args.incar)
    inputset.parameter_checker()
    # check if the parameter is good

    inputset.load_occ(verbose=args.verbose)
    # step 1 initialize global occupation and conditions
    kmc = KMC()
    events_initialized = kmc.initialization(**inputset._parameters)  # v in 10^13 hz

    # # step 2 compute the site kernal (used for kmc run)
    kmc.load_site_event_list(inputset._parameters["event_kernel"])

    # # step 3 run kmc
    kmc.run_from_database(events=events_initialized, **inputset._parameters)

    pass


if __name__ == "__main__":

    main()
