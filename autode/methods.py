from autode.G09 import G09
from autode.XTB import XTB
from autode.config import Config
from autode.exceptions import MethodUnavailable
from autode.log import logger

"""
Functions to get the high and low level electronic structure methods to use for example high-level methods would be
orca and Gaussian09 which can perform DFT/WF theory calculations, low level methods are for example xtb and mopac which
are non ab-initio methods and are therefore considerably faster
"""

high_level_method_names = ['g09']
low_level_method_names = ['xtb']


def get_hmethod():
    """Get the high-level electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method
    """
    # orca = ORCA()
    # g09 = G09()
    # nwchem = NWChem()
    # g16 = G16()
    all_methods = [XTB(), G09()]
    
    return get_defined_method(name=Config.hcode.lower(),
                                  possibilities=all_methods)


def get_lmethod():
    """Get the low-level electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod):
    """
    all_methods = [XTB(), G09()]

    return get_defined_method(name=Config.lcode.lower(),
                                  possibilities=all_methods)



def get_defined_method(name, possibilities):
    """
    Get an electronic structure method defined by it's name

    Arguments:
        name (str):
        possibilities (list(autode.wrappers.base.ElectronicStructureMethod)):

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method
    """

    for method in possibilities:
        if method.name == name:

            method.set_availability()
            # if method.available:
            if True:
                return method

            else:
                logger.critical('Electronic structure method is not available')
                raise MethodUnavailable

    logger.critical('Requested electronic structure code doesn\'t exist')
    raise MethodUnavailable
