%(info_lines)s

import cmath

class ParamCard(object):
    """ Accessor for a SLHA param card.dat."""

    def __init__(self, param_card_path=None):

        if param_card_path is not None:
            raise NotImplementedError(
                "The feature of loading independent parameter values "
                "from a param_card.dat file is not implemented yet.")

        self.set_val = {}

    def get_block_entry(self, block_name, entry_id, default_value):

        if (block_name,entry_id) in self.set_val.keys():
            return self.set_val[ (block_name,entry_id) ]

        # In a future version we will retrieve this value from the param card.
        # For now simply always return the default value.
        return default_value

    def set_block_entry(self, block_name, entry_id, set_value):
        self.set_val[ (block_name,entry_id) ] = set_value
        return

class ModelParameters(object):
    """ This class contains the list of parameters of a physics model %(model_name)s and their definition."""

    def __init__(self, param_card=None):
        """ Instantiates using default value or the path of a SLHA param card."""
       
        # Param card accessor
        if isinstance(param_card, ParamCard):
            slha = param_card
        else:
            slha = ParamCard(param_card)
        
        self.ZERO = 0.

        self.params = {}

        # Computing independent parameters
%(set_independent_parameters)s

        # Computing independent couplings
%(set_independent_couplings)s

        # Computing dependent parameters
%(set_dependent_parameters)s

        # Computing independent parameters
%(set_dependent_couplings)s



        # Setting independent parameters
%(independent_parameters)s

        # Setting independent couplings
%(independent_couplings)s

        # Setting dependent parameters
%(dependent_parameters)s

        # Setting independent parameters
%(dependent_couplings)s



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

    def __str__(self):
        """ Print all parameters contained in this model."""
    
        res = ['>>> Model %(model_name)s <<<']
        res.append('')
        res.append('Independent parameters:')
        res.append('-----------------------')
        res.append('')
%(print_independent_parameters)s

        res.append('')
        res.append('Independent couplings:')
        res.append('----------------------')
        res.append('')
%(print_independent_couplings)s

        res.append('')
        res.append('Dependent parameters:')
        res.append('---------------------')
        res.append('')
%(print_dependent_parameters)s

        res.append('')
        res.append('Dependent couplings:')
        res.append('--------------------')
        res.append('')
%(print_dependent_couplings)s

        res.append('')


        return '\n'.join(res)


