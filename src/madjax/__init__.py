"""The madjax package."""
import jax
import importlib
from madjax.phasespace.flat_phase_space_generator import FlatInvertiblePhasespace


class MadJax:
    def __init__(self, config_name):
        all_processes = importlib.import_module(
            f'{config_name}.processes.all_processes'
        )
        self.parameters = importlib.import_module(f'{config_name}.model.parameters')
        self.processes = {
            k: v for k, v in all_processes.__dict__.items() if 'Matrix_' in k
        }

    def phasespace_generator(self, E_cm, process_name):
        def func(external_parameters):
            parameters = self.parameters.calculate_full_parameters(external_parameters)
            process = self.processes[process_name]()
            external_masses = process.get_external_masses(parameters)
            ps_generator = FlatInvertiblePhasespace(
                external_masses[0],
                external_masses[1],
                beam_Es=(E_cm / 2.0, E_cm / 2.0),
                beam_types=(0, 0),
            )
            # Ensure that E_cm offers enough twice as much energy as necessary
            # to produce the final states
            assert E_cm > sum(external_masses[1]) * 2.0

            return ps_generator
        return func

    
    def jacobian(self, E_cm, process_name, do_jit=True):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            ps_generator = ps(external_parameters)
            PS_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            return jacobian
        return jax.jit(func) if do_jit else func


    def phasespace_vectors(self, E_cm, process_name):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            ps_generator = ps(external_parameters)
            ps_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            return jax.numpy.array([v.vector for v in ps_point])
        return func

    def matrix_element(self, E_cm, process_name, return_grad=True, do_jit=True):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            parameters = self.parameters.calculate_full_parameters(external_parameters)
            ps_generator = ps(external_parameters)
            ps_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            process = self.processes[process_name]()
            return process.smatrix(ps_point, parameters)

        if return_grad:
            return jax.jit(jax.value_and_grad(func)) if do_jit else jax.value_and_grad(func)
        else:
            return jax.jit(func) if do_jit else func


    def matrix_element_and_jacobian(self, E_cm, process_name):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            parameters = self.parameters.calculate_full_parameters(external_parameters)
            ps_generator = ps(external_parameters)
            ps_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            process = self.processes[process_name]()
            return process.smatrix(ps_point, parameters), jacobian

        return func
