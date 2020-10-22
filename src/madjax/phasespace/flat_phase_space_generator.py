import jax
import logging
import math
from .vectors import LorentzVector


class InvalidCmd(RuntimeError):
    pass


class PhaseSpaceGeneratorError(RuntimeError):
    pass


logger = logging.getLogger("MG5aMC_PythonMEs.PhaseSpaceGenerator")


class Dimension(object):
    """ A dimension object specifying a specific integration dimension."""

    def __init__(self, name, folded=False):
        self.name = name
        self.folded = folded

    def length(self):
        raise NotImplementedError()

    def random_sample(self):
        raise NotImplementedError()


class DiscreteDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""

    def __init__(self, name, values, **opts):
        try:
            self.normalized = opts.pop("normalized")
        except:
            self.normalized = False
        super(DiscreteDimension, self).__init__(name, **opts)
        assert isinstance(values, list)
        self.values = values

    def length(self):
        if self.normalized:
            return 1.0 / float(len(self.values))
        else:
            return 1.0

    def random_sample(self):
        return jax.numpy.np.int64(random.choice(self.values))


class ContinuousDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""

    def __init__(self, name, lower_bound=0.0, upper_bound=1.0, **opts):
        super(ContinuousDimension, self).__init__(name, **opts)
        assert upper_bound > lower_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def length(self):
        return self.upper_bound - self.lower_bound

    def random_sample(self):
        return jax.numpy.float64(
            self.lower_bound + random.random() * (self.upper_bound - self.lower_bound)
        )


class DimensionList(list):
    """A DimensionList."""

    def __init__(self, *args, **opts):
        super(DimensionList, self).__init__(*args, **opts)

    def volume(self):
        """ Returns the volue of the complete list of dimensions."""
        vol = 1.0
        for d in self:
            vol *= d.length()
        return vol

    def append(self, arg, **opts):
        """ Type-checking. """
        assert isinstance(arg, Dimension)
        super(DimensionList, self).append(arg, **opts)

    def get_discrete_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, DiscreteDimension))

    def get_continuous_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, ContinuousDimension))

    def random_sample(self):
        return jax.numpy.array([d.random_sample() for d in self])


# =========================================================================================
# Phase space generation
# =========================================================================================


class VirtualPhaseSpaceGenerator(object):
    def __init__(
        self,
        initial_masses,
        final_masses,
        beam_Es,
        beam_types=(1, 1),
        is_beam_factorization_active=(False, False),
        correlated_beam_convolution=False,
    ):

        self.initial_masses = initial_masses
        self.masses = final_masses
        self.n_initial = len(initial_masses)
        self.n_final = len(final_masses)
        self.beam_Es = beam_Es
        self.collider_energy = sum(beam_Es)
        self.beam_types = beam_types
        self.is_beam_factorization_active = is_beam_factorization_active
        self.correlated_beam_convolution = correlated_beam_convolution
        # Sanity check
        if self.correlated_beam_convolution and self.is_beam_factorization_active != (
            True,
            True,
        ):
            raise PhaseSpaceGeneratorError(
                "The beam convolution cannot be set to be correlated if it is one-sided only"
            )
        self.dimensions = self.get_dimensions()
        self.dim_ordered_names = [d.name for d in self.dimensions]

        self.dim_name_to_position = dict(
            (d.name, i) for i, d in enumerate(self.dimensions)
        )
        self.position_to_dim_name = dict(
            (v, k) for (k, v) in self.dim_name_to_position.items()
        )

    def generateKinematics(self, E_cm, random_variables):
        """Generate a phase-space point with fixed center of mass energy."""

        raise NotImplementedError

    def get_PS_point(self, random_variables):
        """Generate a complete PS point, including Bjorken x's,
        dictating a specific choice of incoming particle's momenta."""

        raise NotImplementedError

    def boost_to_lab_frame(self, PS_point, xb_1, xb_2):
        """Boost a phase-space point from the COM-frame to the lab frame, given Bjorken x's."""

        if self.n_initial == 2 and (xb_1 != 1.0 or xb_2 != 1.0):
            ref_lab = PS_point[0] * xb_1 + PS_point[1] * xb_2
            if ref_lab.rho2() != 0.0:
                lab_boost = ref_lab.boostVector()
                for p in PS_point:
                    p.boost(-lab_boost)

    def boost_to_COM_frame(self, PS_point):
        """Boost a phase-space point from the lab frame to the COM frame"""

        if self.n_initial == 2:
            ref_com = PS_point[0] + PS_point[1]
            if ref_com.rho2() != 0.0:
                com_boost = ref_com.boostVector()
                for p in PS_point:
                    p.boost(-com_boost)

    def nDimPhaseSpace(self):
        """Return the number of random numbers required to produce
        a given multiplicity final state."""

        if self.n_final == 1:
            return 0
        return 3 * self.n_final - 4

    def get_dimensions(self):
        """Generate a list of dimensions for this integrand."""

        dims = DimensionList()

        # Add the PDF dimensions if necessary
        if self.beam_types[0] == self.beam_types[1] == 1:
            dims.append(ContinuousDimension("ycms", lower_bound=0.0, upper_bound=1.0))
            # The 2>1 topology requires a special treatment
            if not (self.n_initial == 2 and self.n_final == 1):
                dims.append(
                    ContinuousDimension("tau", lower_bound=0.0, upper_bound=1.0)
                )

        # Add xi beam factorization convolution factors if necessary
        if self.correlated_beam_convolution:
            # A single convolution factor xi that applies to both beams is needed in this case
            dims.append(ContinuousDimension("xi", lower_bound=0.0, upper_bound=1.0))
        else:
            if self.is_beam_factorization_active[0]:
                dims.append(
                    ContinuousDimension("xi1", lower_bound=0.0, upper_bound=1.0)
                )
            if self.is_beam_factorization_active[1]:
                dims.append(
                    ContinuousDimension("xi2", lower_bound=0.0, upper_bound=1.0)
                )

        # Add the phase-space dimensions
        dims.extend(
            [
                ContinuousDimension("x_%d" % i, lower_bound=0.0, upper_bound=1.0)
                for i in range(1, self.nDimPhaseSpace() + 1)
            ]
        )

        return dims


class FlatInvertiblePhasespace(VirtualPhaseSpaceGenerator):
    """Implementation following S. Platzer, arxiv:1308.2922"""

    # This parameter defines a thin layer around the boundary of the unit hypercube
    # of the random variables generating the phase-space,
    # so as to avoid extrema which are an issue in most PS generators.
    epsilon_border = 1e-10

    # The lowest value that the center of mass energy can take.
    # We take here 1 GeV, as anyway below this non-perturbative effects dominate
    # and factorization does not make sense anymore
    absolute_Ecm_min = 1.0

    # For reference here we put the flat weights that Simon uses in his
    # Herwig implementation. I will remove them once I will have understood
    # why they don't match the physical PS volume.
    # So these are not used for now, and get_flatWeights() is used instead.
    flatWeights = {
        2: 0.039788735772973833942,
        3: 0.00012598255637968550463,
        4: 1.3296564302788840628e-7,
        5: 7.0167897579949011130e-11,
        6: 2.2217170114046130768e-14,
    }

    def __init__(self, *args, **opts):
        super(FlatInvertiblePhasespace, self).__init__(*args, **opts)
        if self.n_initial == 1:
            raise InvalidCmd("This basic generator does not support decay topologies.")

    def get_dimensions(self):
        """ Make sure the collider setup is supported."""

        # Check if the beam configuration is supported
        if (not abs(self.beam_types[0]) == abs(self.beam_types[1]) == 1) and (
            not self.beam_types[0] == self.beam_types[1] == 0
        ):
            raise InvalidCmd(
                "This basic generator does not support the collider configuration: (lpp1=%d, lpp2=%d)"
                % (self.run_card["lpp1"], self.run_card["lpp2"])
            )

        if self.beam_Es[0] != self.beam_Es[1]:
            raise InvalidCmd(
                "This basic generator only supports colliders with incoming beams equally energetic."
            )

        return super(FlatInvertiblePhasespace, self).get_dimensions()

    @staticmethod
    def get_flatWeights(E_cm, n, mass=None):
        """Return the phase-space volume for a n massless final states.
        Vol(E_cm, n) = (pi/2)^(n-1) *  (E_cm^2)^(n-2) / ((n-1)!*(n-2)!)
        """
        if n == 1:
            # The jacobian from \delta(s_hat - m_final**2) present in 2->1 convolution
            # must typically be accounted for in the MC integration framework since we
            # don't have access to that here, so we just return 1.
            return 1.0

        return jax.numpy.power((math.pi / 2.0), n - 1) * (
            jax.numpy.power((E_cm ** 2), n - 2)
            / (math.factorial(n - 1) * math.factorial(n - 2))
        )

    @staticmethod
    def bisect(v, n):
        def scalar_solve(f, y):
            return y / f(1.0)

        def binary_search(func, x0, low=0.0, high=1.0, tolerance=1e-6):
            del x0  # unused

            def cond(state):
                low, high = state
                return high - low > tolerance

            def body(state):
                low, high = state
                midpoint = 0.5 * (low + high)
                update_upper = func(midpoint) > 0
                low = jax.numpy.where(update_upper, low, midpoint)
                high = jax.numpy.where(update_upper, midpoint, high)
                return (low, high)

            solution, _ = jax.lax.while_loop(cond, body, (low, high))
            return solution

        tangent_solve = scalar_solve

        def f(u):
            return (u ** (n + 1)) * (n + 2.0 - (n + 1.0) * u) - v

        return jax.lax.custom_root(f, 0.5, binary_search, tangent_solve)

    @staticmethod
    def rho(M, N, m):
        """Returns sqrt((sqr(M)-sqr(N+m))*(sqr(M)-sqr(N-m)))/(8.*sqr(M))"""

        Msqr = M ** 2
        return ((Msqr - (N + m) ** 2) * (Msqr - (N - m) ** 2)) ** 0.5 / (8.0 * Msqr)

    def setInitialStateMomenta(self, output_momenta, E_cm):
        """Generate the initial state momenta."""

        if self.n_initial not in [1, 2]:
            raise InvalidCmd("This PS generator only supports 1 or 2 initial states")

        if self.n_initial == 1:
            if self.initial_masses[0] == 0.0:
                raise PhaseSpaceGeneratorError(
                    "Cannot generate the decay phase-space of a massless particle."
                )
            if self.E_cm != self.initial_masses[0]:
                raise PhaseSpaceGeneratorError(
                    "Can only generate the decay phase-space of a particle at rest."
                )

        if self.n_initial == 1:
            output_momenta[0] = LorentzVector([self.initial_masses[0], 0.0, 0.0, 0.0])
            return

        elif self.n_initial == 2:
            if self.initial_masses[0] == 0.0 or self.initial_masses[1] == 0.0:
                output_momenta[0] = LorentzVector(
                    [E_cm / 2.0, 0.0, 0.0, 1.0 * E_cm / 2.0]
                )
                output_momenta[1] = LorentzVector(
                    [E_cm / 2.0, 0.0, 0.0, (-1.0) * E_cm / 2.0]
                )
            else:
                M1sq = self.initial_masses[0] ** 2
                M2sq = self.initial_masses[1] ** 2
                E1 = (E_cm ** 2 + M1sq - M2sq) / E_cm
                E2 = (E_cm ** 2 - M1sq + M2sq) / E_cm
                Z = (
                    jax.numpy.sqrt(
                        E_cm ** 4
                        - 2 * E_cm ** 2 * M1sq
                        - 2 * E_cm ** 2 * M2sq
                        + M1sq ** 2
                        - 2 * M1sq * M2sq
                        + M2sq ** 2
                    )
                    / E_cm
                )
                output_momenta[0] = LorentzVector([E1 / 2.0, 0.0, 0.0, 1.0 * Z / 2.0])
                output_momenta[1] = LorentzVector(
                    [E2 / 2.0, 0.0, 0.0, (-1.0) * Z / 2.0]
                )
        return

    def get_PS_point(self, random_variables):
        """Generate a complete PS point, including Bjorken x's,
        dictating a specific choice of incoming particle's momenta.
        """

        # if random_variables are not defined, than just throw a completely random point
        if random_variables is None:
            random_variables = self.dimensions.random_sample()

        # Check the sensitivity of te inputs from the integrator
        # Lukas: not part of computation ignore for now
        # if jax.numpy.any(jax.numpy.isnan(random_variables)):
        #     logger.warning('Some input variables from the integrator are malformed: %s'%
        #         ( ', '.join( '%s=%s'%( name, random_variables[pos]) for name, pos in
        #                                              self.dim_name_to_position.items() ) ))
        #     logger.warning('The PS generator will yield None, triggering the point to be skipped.')
        #     return None, 0.0, (0., 0.), (0., 0.)

        # Phase-space point weight to return
        wgt = 1.0

        # if any(math.isnan(r) for r in random_variables):
        #    misc.sprint(random_variables)

        # Avoid extrema since the phase-space generation algorithm doesn't like it
        random_variables = [
            min(max(rv, self.epsilon_border), 1.0 - self.epsilon_border)
            for rv in random_variables
        ]

        # Assign variables to their meaning.
        if "ycms" in self.dim_name_to_position:
            PDF_ycm = random_variables[self.dim_name_to_position["ycms"]]
        else:
            PDF_ycm = None
        if "tau" in self.dim_name_to_position:
            PDF_tau = random_variables[self.dim_name_to_position["tau"]]
        else:
            PDF_tau = None
        PS_random_variables = [
            rv
            for i, rv in enumerate(random_variables)
            if self.position_to_dim_name[i].startswith("x_")
        ]

        # Also generate the ISR collinear factorization convolutoin variables xi<i> if
        # necessary. In order for the + distributions of the PDF counterterms and integrated
        # collinear ISR counterterms to hit the PDF only (and not the matrix elements or
        # observables functions), a change of variable is necessary: xb_1' = xb_1 * xi1
        if self.correlated_beam_convolution:
            # Both xi1 and xi2 must be set equal then
            xi1 = random_variables[self.dim_name_to_position["xi"]]
            xi2 = random_variables[self.dim_name_to_position["xi"]]
        else:
            if self.is_beam_factorization_active[0]:
                xi1 = random_variables[self.dim_name_to_position["xi1"]]
            else:
                xi1 = None
            if self.is_beam_factorization_active[1]:
                xi2 = random_variables[self.dim_name_to_position["xi2"]]
            else:
                xi2 = None

        # Now take care of the Phase-space generation:
        # Set some defaults for the variables to be set further
        xb_1 = 1.0
        xb_2 = 1.0
        E_cm = self.collider_energy

        # We generate the PDF from two variables \tau = x1*x2 and ycm = 1/2 * log(x1/x2), so that:
        #  x_1 = sqrt(tau) * exp(+ycm)
        #  x_2 = sqrt(tau) * exp(-ycm)
        # The jacobian of this transformation is 1.
        if abs(self.beam_types[0]) == abs(self.beam_types[1]) == 1:

            tot_final_state_masses = sum(self.masses)
            if tot_final_state_masses > self.collider_energy:
                raise PhaseSpaceGeneratorError(
                    "Collider energy is not large enough, there is no phase-space left."
                )

            # Keep a hard cut at 1 GeV, which is the default for absolute_Ecm_min
            tau_min = (
                max(tot_final_state_masses, self.absolute_Ecm_min)
                / self.collider_energy
            ) ** 2
            tau_max = 1.0

            if self.n_initial == 2 and self.n_final == 1:
                # Here tau is fixed by the \delta(xb_1*xb_2*s - m_h**2) which sets tau to
                PDF_tau = tau_min
                # Account for the \delta(xb_1*xb_2*s - m_h**2) and corresponding y_cm matching to unit volume
                wgt *= 1.0 / self.collider_energy ** 2
            else:
                # Rescale tau appropriately
                PDF_tau = tau_min + (tau_max - tau_min) * PDF_tau
                # Including the corresponding Jacobian
                wgt *= tau_max - tau_min

            # And we can now rescale ycm appropriately
            ycm_min = 0.5 * jax.numpy.log(PDF_tau)
            ycm_max = -ycm_min
            PDF_ycm = ycm_min + (ycm_max - ycm_min) * PDF_ycm
            # and account for the corresponding Jacobian
            wgt *= ycm_max - ycm_min

            xb_1 = jax.numpy.sqrt(PDF_tau) * jax.numpy.exp(PDF_ycm)
            xb_2 = jax.numpy.sqrt(PDF_tau) * jax.numpy.exp(-PDF_ycm)
            # /!\ The mass of initial state momenta is neglected here.
            E_cm = jax.numpy.sqrt(xb_1 * xb_2) * self.collider_energy

        elif self.beam_types[0] == self.beam_types[1] == 0:
            xb_1 = 1.0
            xb_2 = 1.0
            E_cm = self.collider_energy
        else:
            raise InvalidCmd(
                "This basic PS generator does not yet support collider mode (%d,%d)."
                % self.beam_types
            )

        # Now generate a PS point
        PS_point, PS_weight = self.generateKinematics(E_cm, PS_random_variables)

        # Apply the phase-space weight
        wgt *= PS_weight

        return PS_point, wgt, (xb_1, xi1), (xb_2, xi2)

    def generateKinematics(self, E_cm, random_variables):
        """Generate a self.n_initial -> self.n_final phase-space point
        using the random variables passed in argument.
        """

        # Make sure the right number of random variables are passed
        try:
            assert len(random_variables) == self.nDimPhaseSpace()
        except:
            raise RuntimeError('need {} random variables'.format(self.nDimPhaseSpace()))

        # Make sure that none of the random_variables is NaN.
        # Lukas: not part of computation ignore for now
        # if jax.numpy.any(jax.numpy.isnan(random_variables)):
        #     raise PhaseSpaceGeneratorError("Some of the random variables passed "+
        #       "to the phase-space generator are NaN: %s"%str(random_variables))

        # The distribution weight of the generate PS point
        weight = 1.0

        output_momenta = []

        mass = self.masses[0]
        if self.n_final == 1:
            if self.n_initial == 1:
                raise InvalidCmd("1 > 1 phase-space generation not supported.")
            if mass / E_cm < 1.0e-7 or ((E_cm - mass) / mass) > 1.0e-7:
                raise PhaseSpaceGeneratorError(
                    "1 > 2 phase-space generation needs a final state mass equal to E_c.o.m."
                )
            output_momenta.append(
                LorentzVector([mass / 2.0, 0.0, 0.0, 1.0 * mass / 2.0])
            )
            output_momenta.append(
                LorentzVector([mass / 2.0, 0.0, 0.0, (-1.0) * mass / 2.0])
            )
            output_momenta.append(LorentzVector([mass, 0.0, 0.0, 0.0]))
            weight = self.get_flatWeights(E_cm, 1)
            return output_momenta, weight

        M = [0.0] * (self.n_final - 1)
        M[0] = E_cm

        weight *= self.generateIntermediatesMassive(M, E_cm, random_variables)
        M.append(self.masses[-1])

        Q = LorentzVector([M[0], 0.0, 0.0, 0.0])
        nextQ = LorentzVector()

        for i in range(self.n_initial + self.n_final - 1):

            if i < self.n_initial:
                output_momenta.append(LorentzVector())
                continue

            q = (
                4.0
                * M[i - self.n_initial]
                * self.rho(
                    M[i - self.n_initial],
                    M[i - self.n_initial + 1],
                    self.masses[i - self.n_initial],
                )
            )
            cos_theta = (
                2.0 * random_variables[self.n_final - 2 + 2 * (i - self.n_initial)]
                - 1.0
            )
            sin_theta = jax.numpy.sqrt(1.0 - cos_theta ** 2)
            phi = (
                2.0
                * math.pi
                * random_variables[self.n_final - 1 + 2 * (i - self.n_initial)]
            )
            cos_phi = jax.numpy.cos(phi)
            sin_phi = jax.numpy.sqrt(1.0 - cos_phi ** 2)

            sin_phi = jax.numpy.where(phi > math.pi, (-1.0) * sin_phi, sin_phi)
            # if (phi > math.pi):
            #     sin_phi = -sin_phi

            p = LorentzVector(
                [0.0, q * sin_theta * cos_phi, q * sin_theta * sin_phi, q * cos_theta]
            )

            p.set_square(self.masses[i - self.n_initial] ** 2)
            p.boost(Q.boostVector())
            p.set_square(self.masses[i - self.n_initial] ** 2)
            output_momenta.append(p)

            nextQ = Q - p
            nextQ.set_square(M[i - self.n_initial + 1] ** 2)
            Q = nextQ

        output_momenta.append(Q)

        self.setInitialStateMomenta(output_momenta, E_cm)

        return output_momenta, weight

    def generateIntermediatesMassless(self, M, E_cm, random_variables):
        """Generate intermediate masses for a massless final state."""

        for i in range(2, self.n_final):
            u = self.bisect(random_variables[i - 2], self.n_final - 1 - i)
            M[i - 1] = jax.numpy.sqrt(u * (M[i - 2] ** 2))

        return self.get_flatWeights(E_cm, self.n_final)

    def generateIntermediatesMassive(self, M, E_cm, random_variables):
        """Generate intermediate masses for a massive final state."""

        K = list(M)
        K[0] -= sum(self.masses)

        weight = self.generateIntermediatesMassless(K, E_cm, random_variables)

        del M[:]
        M.extend(K)

        for i in range(1, self.n_final):
            for k in range(i, self.n_final + 1):
                M[i - 1] += self.masses[k - 1]

        weight *= 8.0 * self.rho(
            M[self.n_final - 2],
            self.masses[self.n_final - 1],
            self.masses[self.n_final - 2],
        )
        for i in range(2, self.n_final):
            weight *= (
                self.rho(M[i - 2], M[i - 1], self.masses[i - 2])
                / self.rho(K[i - 2], K[i - 1], 0.0)
            ) * (M[i - 1] / K[i - 1])

        weight *= jax.numpy.power(K[0] / M[0], 2 * self.n_final - 4)
        return weight

    def invertKinematics(self, E_cm, momenta):
        """ Returns the random variables that yields the specified momenta configuration."""

        # Make sure the right number of momenta are passed
        assert len(momenta) == (self.n_initial + self.n_final)
        moms = momenta.copy()

        # The weight of the corresponding PS point
        weight = 1.0

        if self.n_final == 1:
            if self.n_initial == 1:
                raise PhaseSpaceGeneratorError(
                    "1 > 1 phase-space generation not supported."
                )
            return [], self.get_flatWeights(E_cm, 1)

        # The random variables that would yield this PS point.
        random_variables = [-1.0] * self.nDimPhaseSpace()

        M = [
            0.0,
        ] * (self.n_final - 1)
        M[0] = E_cm

        Q = [
            LorentzVector(),
        ] * (self.n_final - 1)
        Q[0] = LorentzVector([M[0], 0.0, 0.0, 0.0])

        for i in range(2, self.n_final):
            for k in range(i, self.n_final + 1):
                Q[i - 1] = Q[i - 1] + moms[k + self.n_initial - 1]
            M[i - 1] = abs(Q[i - 1].square()) ** 0.5

        weight = self.invertIntermediatesMassive(M, E_cm, random_variables)

        for i in range(self.n_initial, self.n_final + 1):
            # BALDY another copy? moms not used afterwards
            p = LorentzVector(moms[i])
            # Take the opposite boost vector
            boost_vec = -1.0 * Q[i - self.n_initial].boostVector()
            p.boost(boost_vec)
            random_variables[self.n_final - 2 + 2 * (i - self.n_initial)] = (
                p.cosTheta() + 1.0
            ) / 2.0
            phi = p.phi()
            if phi < 0.0:
                phi += 2.0 * math.pi
            random_variables[self.n_final - 1 + 2 * (i - self.n_initial)] = phi / (
                2.0 * math.pi
            )

        return random_variables, weight

    def invertIntermediatesMassive(self, M, E_cm, random_variables):
        """ Invert intermediate masses for a massive final state."""

        K = list(M)
        for i in range(1, self.n_final):
            K[i - 1] -= sum(self.masses[i - 1 :])

        weight = self.invertIntermediatesMassless(K, E_cm, random_variables)
        weight *= 8.0 * self.rho(
            M[self.n_final - 2],
            self.masses[self.n_final - 1],
            self.masses[self.n_final - 2],
        )
        for i in range(2, self.n_final):
            weight *= (
                self.rho(M[i - 2], M[i - 1], self.masses[i - 2])
                / self.rho(K[i - 2], K[i - 1], 0.0)
            ) * (M[i - 1] / K[i - 1])

        weight *= jax.numpy.power(K[0] / M[0], 2 * self.n_final - 4)

        return weight

    def invertIntermediatesMassless(self, K, E_cm, random_variables):
        """ Invert intermediate masses for a massless final state."""

        for i in range(2, self.n_final):
            u = (K[i - 1] / K[i - 2]) ** 2
            random_variables[i - 2] = (self.n_final + 1 - i) * jax.numpy.power(
                u, self.n_final - i
            ) - (self.n_final - i) * jax.numpy.power(u, self.n_final + 1 - i)

        return self.get_flatWeights(E_cm, self.n_final)


# =========================================================================================
# Standalone main for debugging / standalone trials
# =========================================================================================
if __name__ == "__main__":

    import random

    E_cm = 5000.0

    # Try to run the above for a 2->8.
    my_PS_generator = FlatInvertiblePhasespace(
        [0.0] * 2,
        [100.0 + 10.0 * i for i in range(2)],
        beam_Es=(E_cm / 2.0, E_cm / 2.0),
        beam_types=(0, 0),
    )
    # Try to run the above for a 2->1.
    #    my_PS_generator = FlatInvertiblePhasespace([0.]*2, [5000.0])

    random_variables = [
        random.random() for _ in range(my_PS_generator.nDimPhaseSpace())
    ]

    momenta, wgt = my_PS_generator.generateKinematics(E_cm, random_variables)

    print("\nRandom variables :\n", random_variables)
    print("\n%s\n" % momenta.__str__(n_initial=my_PS_generator.n_initial))
    print("Phase-space weight : %.16e\n" % wgt)

    variables_reconstructed, wgt_reconstructed = my_PS_generator.invertKinematics(
        E_cm, momenta
    )

    print("\n =========================")
    print(" || Kinematic inversion ||")
    print(" =========================")
    print("\nReconstructed random variables :\n", variables_reconstructed)
    differences = [
        abs(variables_reconstructed[i] - random_variables[i])
        for i in range(len(variables_reconstructed))
    ]
    print("Reconstructed weight = %.16e" % wgt_reconstructed)
    if differences:
        print(
            "\nMax. relative diff. in reconstructed variables = %.3e"
            % max(differences[i] / random_variables[i] for i in range(len(differences)))
        )
    print("Rel. diff. in PS weight = %.3e\n" % ((wgt_reconstructed - wgt) / wgt))

    print(("-" * 100))
    print(("SIDE EXPERIMENT, TO REMOVE LATER"))
    print(("-" * 100))
    import copy

    my_PS_generator = FlatInvertiblePhasespace(
        [0.0] * 2,
        [100.0 + 10.0 * i for i in range(2)],
        beam_Es=(E_cm / 2.0, E_cm / 2.0),
        beam_types=(0, 0),
    )
    # Try to run the above for a 2->1.
    #    my_PS_generator = FlatInvertiblePhasespace([0.]*2, [5000.0])

    random_variables = [
        random.random() for _ in range(my_PS_generator.nDimPhaseSpace())
    ]

    momenta, wgt = my_PS_generator.generateKinematics(E_cm, random_variables)

    print("original momenta")
    print(str(momenta))
    print("computing boost vector to lab frame with x1=0.25 and x2=0.6")
    boost_vector_to_lab_frame = None
    xb_1, xb_2 = 0.25, 0.6
    ref_lab = momenta[0] * xb_1 + momenta[1] * xb_2
    boost_vector_to_lab_frame = -1.0 * ref_lab.boostVector()

    print("then further rescaling with xi1=0.12 and xi2=0.71")
    xi1, xi2 = 0.12, 0.71
    momenta[0] = xi1 * momenta[0]
    momenta[1] = xi2 * momenta[1]

    print("Boosting back to c.o.m:")
    ref_com = momenta[0] + momenta[1]
    xi_boost = ref_com.boostVector()
    boost_vector_to_lab_frame += xi_boost
    momenta_boosted_to_com = copy.deepcopy(momenta)
    for p in momenta_boosted_to_com:
        p.boost(-1.0 * xi_boost)
    test_momenta = copy.deepcopy(momenta_boosted_to_com)

    print("Final PS point should be in c.o.m:")
    print(str(momenta_boosted_to_com))
    print("\nFinal check\n")
    print("Initial state momenta obtained from simple rescaling:")
    print("1: %s" % str(momenta_boosted_to_com[0] * xb_1 * xi1))
    print("2: %s" % str(momenta_boosted_to_com[1] * xb_2 * xi2))

    print("\n Initial state momenta obtained with xi boost:")
    print("xi_boost=", str(xi_boost))
    test_momenta[0].boost(xi_boost)
    test_momenta[1].boost(xi_boost)
    print("1: %s" % str(test_momenta[0]))
    print("2: %s" % str(test_momenta[1]))
    print(".vs.")
    print("1: %s" % str(momenta[0]))
    print("2: %s" % str(momenta[1]))

    print("\n Initial state momenta obtained with subsequent bjorken boost:")
    test_momenta[0].boost(-1.0 * ref_lab.boostVector())
    test_momenta[1].boost(-1.0 * ref_lab.boostVector())
    print("1: %s" % str(test_momenta[0]))
    print("2: %s" % str(test_momenta[1]))
    print(".vs.")
    print("1: %s" % str(momenta[0] * xb_1))
    print("2: %s" % str(momenta[1] * xb_2))

    print("\n Initial state momenta obtained with overall boost:")
    momenta[0].boost(boost_vector_to_lab_frame)
    momenta[1].boost(boost_vector_to_lab_frame)
    print("1: %s" % str(momenta[0]))
    print("2: %s" % str(momenta[1]))
