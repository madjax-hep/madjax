import jax
import importlib
import random
import os
import sys
from timeit import default_timer as timer
from madjax.phasespace.flat_phase_space_generator import FlatInvertiblePhasespace

setup_name = sys.argv[1]

all_processes = importlib.import_module('{}.processes.all_processes'.format(setup_name))
parameters = importlib.import_module('{}.model.parameters'.format(setup_name))
all_process_classes = [v for k, v in all_processes.__dict__.items() if 'Matrix_' in k]

# For now, the feature of specifying an SLHA param card to initialise
# the value of independent parameters is not supported yet.
active_model = parameters.ModelParameters(None)

# Center of mass of the collision in GeV
E_cm = 14000.0

for process_class in all_process_classes:

    print(">>> Running process ", process_class)

    # Generate a random PS point for this process
    process = process_class()
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

    # Generate some random variables
    random_variables = [random.random() for _ in range(ps_generator.nDimPhaseSpace())]

    def matrix_element(c, PS_point):
        PS_point, jacobian = ps_generator.generateKinematics(
            this_process_E_cm, random_variables
        )
        _Z_mass = c

        _pc = parameters.ParamCard()
        _pc.set_block_entry("mass", 23, _Z_mass)  # 9.118800e+01
        _active_model_params = parameters.ModelParameters(_pc).params

        _process = process_class()
        return process.smatrix(PS_point, _active_model_params)

    matrix_element_jit = jax.jit(matrix_element)
    matrix_element_prime = jax.grad(matrix_element, 0)
    matrix_element_prime_jit = jax.jit(matrix_element_prime)

    print("ME:", matrix_element(9.918800e01, random_variables))
    start = timer()
    print("ME jit:", matrix_element_jit(9.918800e01, random_variables))
    end = timer()
    print("ME jit compilation + eval time:", (end - start))

    print("ME prime:", matrix_element_prime(9.918800e01, random_variables))
    start = timer()
    print("ME prime jit:", matrix_element_prime_jit(9.918800e01, random_variables))
    end = timer()
    print("ME primt jit compilation + eval time:", (end - start))

    # -----------------
    # Timing Test
    # -----------------
    start = timer()
    matrix_element(9.818800e01, random_variables)
    matrix_element(9.718800e01, random_variables)
    end = timer()
    print("ME ave time:", (end - start) / 2.0)

    start = timer()
    matrix_element_jit(9.818800e01, random_variables)
    matrix_element_jit(9.718800e01, random_variables)
    end = timer()
    print("ME jit ave time:", (end - start) / 2.0)

    start = timer()
    matrix_element_prime(9.818800e01, random_variables)
    matrix_element_prime(9.718800e01, random_variables)
    end = timer()
    print("ME prime ave time:", (end - start) / 2.0)

    start = timer()
    matrix_element_prime_jit(9.818800e01, random_variables)
    matrix_element_prime_jit(9.718800e01, random_variables)
    end = timer()
    print("ME prime ave time:", (end - start) / 2.0)

    # -----------------
    # Only eval one ME
    # -----------------
    break
