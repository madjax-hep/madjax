from madjax.phasespace.flat_phase_space_generator import FlatInvertiblePhasespace
import random
import numpy as np
def test_closure():
    external_masses = [[5.0,5.0] ,[0.0,0.0]]
    E_cm = 125.0
    ps_generator = FlatInvertiblePhasespace(
                external_masses[0],
                external_masses[1],
                beam_Es=(E_cm / 2.0, E_cm / 2.0),
                beam_types=(0, 0),
            )

    random_variables = np.array([random.random() for i in range(2)])
    PS_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
    print(random_variables)
              
    rvs, weight = ps_generator.invertKinematics(E_cm, PS_point)
    print(rvs)




