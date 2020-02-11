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
        def func(external_parameters, random_variables):
            # Generate a random PS point for this process
            parameters = self.parameters.calculate_full_parameters(external_parameters)

            process = self.processes[process_name]()
            external_masses = process.get_external_masses(parameters)

            # Ensure that E_cm offers enough twice as much energy as necessary
            # to produce the final states
            assert E_cm > sum(external_masses[1]) * 2.0

            ps_generator = FlatInvertiblePhasespace(
                external_masses[0],
                external_masses[1],
                beam_Es=(E_cm / 2.0, E_cm / 2.0),
                beam_types=(0, 0),
            )

            PS_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            return process.smatrix(PS_point, parameters)

        if return_grad:
            return jax.jit(jax.value_and_grad(func))
        else:
            return jax.jit(func)
