import random
import sys
import madjax

mj = madjax.MadJax(config_name=sys.argv[1])
for k, process_class in mj.processes.items():
    print(">>> Running process ", k)

    # Generate some random variables
    nDimPhaseSpace = 2
    random_variables = [0.2, 0.3]

    # Center of mass of the collision in GeV
    matrix_element = mj.matrix_element(E_cm=14000.0, process_name=k)

    v, p = matrix_element(9.918800e01, random_variables)
    print("ME:", v)
    print("ME prime:", p)
