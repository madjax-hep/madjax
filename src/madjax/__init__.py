"""The madjax package."""
import jax
import importlib
from madjax.phasespace.flat_phase_space_generator import FlatInvertiblePhasespace


class MadJax(object):
    def __init__(self, config_name):
        all_processes = importlib.import_module(
            '{}.processes.all_processes'.format(config_name)
        )
        self.parameters = importlib.import_module(
            '{}.model.parameters'.format(config_name)
        )
        self.processes = {
            k: v for k, v in all_processes.__dict__.items() if 'Matrix_' in k
        }

    def matrix_element(self, E_cm, process_name, return_grad=True):
        def func(parameter, random_variables):
            # Generate a random PS point for this process
            _Z_mass = parameter
            _pc = self.parameters.ParamCard()
            _pc.set_block_entry("mass", 23, _Z_mass)  # 9.118800e+01
            active_model = self.parameters.ModelParameters(_pc)

            process = self.processes[process_name]()
            external_masses = process.get_external_masses(active_model)

            # Ensure that E_cm offers enough twice as much energy as necessary
            # to produce the final states
            this_process_E_cm = max(E_cm, sum(external_masses[1]) * 2.0)

            ps_generator = FlatInvertiblePhasespace(
                external_masses[0],
                external_masses[1],
                beam_Es=(this_process_E_cm / 2.0, this_process_E_cm / 2.0),
                # We do not consider PDF for this standalone check
                beam_types=(0, 0),
            )

            PS_point, jacobian = ps_generator.generateKinematics(
                this_process_E_cm, random_variables
            )
            return process.smatrix(PS_point, active_model.params)

        if return_grad:
            return jax.jit(jax.value_and_grad(func))
        else:
            return jax.jit(func)
