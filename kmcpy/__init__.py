"""Main kMCpy module."""
import datetime
import logging
import os

from ._version import __version__

__author__ = "kMCpy Developers"
__author_email__ = "dengzeyu@gmail.com"

# Expose the version and the logo function as the public API for this module.
__all__ = ["__version__", "get_logo"]


# 2. LOGGING SETUP: This is perfect.
logging.getLogger(__name__).addHandler(logging.NullHandler())


# 3. LOGO FUNCTION: A stateless function to generate and print the logo.

def get_logo():
    """
    Prints the kMCpy ASCII logo and version/citation information to the console.

    This function can be called multiple times. It's recommended to call
    it once at the start of an application's main entry point.
    """
    # Ansi color codes for a professional look.
    class Colors:
        CYAN = '\033[96m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        END = '\033[0m'

    # The logo art. Using a raw string (r"...") is good practice.
    logo_art = r"""
  _    __  __  _____             
 | |  |  \/  |/ ____|            
 | | _| \  / | |     _ __  _   _ 
 | |/ / |\/| | |    | '_ \| | | |
 |   <| |  | | |____| |_) | |_| |
 |_|\_\_|  |_|\_____| .__/ \__, |
                    | |     __/ |
                    |_|    |___/ 
"""
    
    # Using os.uname() is not cross-platform (it fails on Windows).
    # platform.node() is a more robust alternative.
    import platform
    hostname = platform.node()

    # Assemble the final message string inside the function.
    # This ensures all information (like datetime) is current.
    message = (
        f"{Colors.CYAN}{Colors.BOLD}{logo_art}{Colors.END}"
        f"kMCpy: A python package to simulate transport properties in solids with kinetic Monte Carlo\n"
        f"{Colors.YELLOW}Please cite:{Colors.END} DOI: 10.1016/j.commatsci.2023.112394 and 10.1038/s41467-022-32190-7 \n"
        f"{Colors.YELLOW}Author:{Colors.END} {__author__} <{__author_email__}>\n"
        f"Version: {__version__}\n"
        f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Host: {hostname}\n"
    )
    
    return message
