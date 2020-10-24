import sys
import madjax

mj = madjax.MadJax(config_name=sys.argv[1])
for process, process_class in mj.processes.items():
    print(f">>> Running process {process}")

    # Generate some random variables
    nDimPhaseSpace = 2
    random_variables = [0.2] * nDimPhaseSpace

    # Center of mass of the collision in GeV
    matrix_element = mj.matrix_element(E_cm=125.0, process_name=process)

    v, p = matrix_element(
        {
            ("mass", 23): 9.918800e01,
            ("sminputs", 2): 1.166390e-05,
            ("mass", 25): 1.250000e02,
        },
        random_variables,
    )
    print(f"ME: {v}")
    print(f"ME prime: {p}")
