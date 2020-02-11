%(info_lines)s

import cmath

def calculate_full_parameters(slha):
        """ Instantiates using default value or the path of a SLHA param card."""
       
        
        _result_params = {'ZERO': 0.0}

        # Computing independent parameters
%(set_independent_parameters)s

        # Computing independent couplings
%(set_independent_couplings)s

        # Computing dependent parameters
%(set_dependent_parameters)s

        # Computing independent parameters
%(set_dependent_couplings)s

        # ------------------------------
        # Building Dictionary
        # ------------------------------
        # Setting independent parameters
%(independent_parameters_dict)s

        # Setting independent couplings
%(independent_couplings_dict)s

        # Setting dependent parameters
%(dependent_parameters_dict)s

        # Setting independent parameters
%(dependent_couplings_dict)s



        return _result_params