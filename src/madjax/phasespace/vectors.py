##########################################################################################
#
# Copyright (c) 2017 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
##########################################################################################

import logging
import math
import copy
import numpy as np

logger = logging.getLogger("madgraph.PhaseSpaceGenerator")


class InvalidOperation(Exception):
    pass


def almost_equal(x, y, rel_tol=0, abs_tol=0):
    """Check if two objects are equal within certain relative and absolute tolerances.
    The operations abs(x + y) and abs(x - y) need to be well-defined
    for this function to work.

    :param x: first object to be compared
    :param y: second object to be compared
    :param rel_tol: relative tolerance for the comparison
    :param abs_tol: absolute tolerance for the comparison
    :return: True if the elements are equal within tolerance, False otherwise
    :rtype: bool
    """

    diffxy = abs(x - y)
    if diffxy <= abs_tol:
        return True
    sumxy = abs(x + y)
    # Rough check that the ratio is smaller than 1 to avoid division by zero
    if sumxy < diffxy:
        return False
    return diffxy / sumxy <= rel_tol


# =========================================================================================
# Vector
# =========================================================================================
import jax.numpy as np
import jax


class What3(object):
    def __init__(self, vec):
        self.vector = np.asarray(vec)

    def __getitem__(self, index):
        return self.vector[index]

    def square(self):

        return self.vector.dot(self.vector)

    def __truediv__(self, scalar):
        return What3(self.vector / scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __mul__(self, scalar):
        return What3(self.vector * scalar)

    def __mull__(self, scalar):
        return What3(self.vector * scalar)


class What(object):
    def __init__(self, vec):
        self.vector = np.asarray(vec)

    def __mul__(self, scalar):
        return What(self.vector * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __getitem__(self, index):
        return self.vector[index]

    def __setitem__(self, index, value):
        self.vector = jax.ops.index_update(self.vector, index, value)

    def __sub__(self, other):
        return What(self.vector - other.vector)

    def __add__(self, other):
        return What(self.vector + other.vector)

    def __len__(self):
        return len(self.vector)

    def asarray(self):
        return self.vector

    def square(self):

        return self.dot(self.vector)

    def set_square(self, square, negative=False):
        """Change the time component of this LorentzVector
        in such a way that self.square() = square.
        If negative is True, set the time component to be negative,
        else assume it is positive.
        """

        # Note: square = self[0]**2 - self.rho2(),
        # so if (self.rho2() + square) is negative, self[0] is imaginary.
        # Letting math.sqrt fail if data is not complex on purpose in this case.
        self[0] = (self.rho2() + square) ** 0.5
        if negative:
            self[0] *= -1
        return self

    def space(self):
        """Return the spatial part of this LorentzVector."""

        return What3(self[1:])

    def dot(self, v, out=None):
        """Compute the Lorentz scalar product."""
        ## The implementation below allows for a check but it should be done upstream and
        ## significantly slows down the code here.
        # pos = self[0]*v[0]
        # neg = self.space().dot(v.space())
        # if pos+neg != 0 and abs(2*(pos-neg)/(pos+neg)) < 100.*self.eps(): return 0
        # return pos - neg
        return self[0] * v[0] - self[1] * v[1] - self[2] * v[2] - self[3] * v[3]

    def square_almost_zero(self):
        """Check if the square of this LorentzVector is zero within numerical accuracy."""

        return self.almost_zero(self.square() / np.dot(self, self))

    def rho2(self):
        """Compute the radius squared."""

        return self.space().square()

    def rho(self):
        """Compute the radius."""

        return abs(self.space())

    def space_direction(self):
        """Compute the corresponding unit vector in ordinary space."""

        return self.space() / self.rho()

    def phi(self):

        return math.atan2(self[2], self[1])

    def boost(self, boost_vector, gamma=-1.0):
        """Transport self into the rest frame of the boost_vector in argument.
        This means that the following command, for any vector p=(E, px, py, pz)
            p.boost(-p.boostVector())
        transforms p to (M,0,0,0).
        """

        b2 = boost_vector.square()
        if gamma < 0.0:
            gamma = 1.0 / jax.numpy.sqrt(1.0 - b2)

        bp = self.space().vector.dot(boost_vector.vector)
        gamma2 = jax.numpy.where(b2 > 0, gamma - 1.0 / b2, 0.0)
        factor = gamma2 * bp + gamma * self[0]
        # self.space().vector += factor*boost_vector.vector
        self[1:] += factor * boost_vector.vector
        self[0] = gamma * (self[0] + bp)
        return self

    def boostVector(self):

        if self == LorentzVector():
            return Vector([0.0] * 3)
        # if self[0] <= 0. or self.square() < 0.:
        #     logger.critical("Attempting to compute a boost vector from")
        #     logger.critical("%s (%.9e)" % (str(self), self.square()))
        #     raise InvalidOperation
        return self.space() / self[0]

    def cosTheta(self):

        ptot = self.rho()
        assert ptot > 0.0
        return self[3] / ptot


class Vector(np.ndarray):
    def __new__(cls, *args, **opts):
        if args and isinstance(args[0], Vector):
            vec = args[0]
            vec = [vec[i] for i in range(len(vec))]

        else:
            vec = args[0]
            vec = [vec[i] for i in range(len(vec))]
        return What(vec)


# =========================================================================================
# LorentzVector
# =========================================================================================


class LorentzVector(Vector):
    def __new__(cls, *args, **opts):
        if len(args) == 0:
            return super(LorentzVector, cls).__new__(cls, [0.0, 0.0, 0.0, 0.0], **opts)
        return super(LorentzVector, cls).__new__(cls, *args, **opts)


# =========================================================================================
# LorentzVectorDict
# =========================================================================================


class LorentzVectorDict(dict):
    """A simple class wrapping dictionaries that store Lorentz vectors."""

    def to_list(self, ordered_keys=None):
        """Return list copy of self. Notice that the actual values of the keys
        are lost in this process. The user can specify in which order (and potentially which ones)
        the keys must be placed in the list returned."""

        if ordered_keys is None:
            return LorentzVectorList(self[k] for k in sorted(self.keys()))
        else:
            return LorentzVectorList(self[k] for k in ordered_keys)

    def to_dict(self):
        """Return a copy of this LorentzVectorDict """

        return LorentzVectorDict(self)

    def to_tuple(self):
        """Return a copy of this LorentzVectorDict as an immutable tuple.
        Notice that the actual values of the keys are lost in this process.
        """

        return tuple(tuple(self[k]) for k in sorted(self.keys()))

    def __str__(self, n_initial=2):
        """Nice printout of the momenta."""

        # Use padding for minus signs
        def special_float_format(fl):
            return "%s%.16e" % ("" if fl < 0.0 else " ", fl)

        cols_widths = [4, 25, 25, 25, 25, 25]
        template = " ".join("%%-%ds" % col_width for col_width in cols_widths)
        line = "-" * (sum(cols_widths) + len(cols_widths) - 1)

        out_lines = [template % ("#", " E", " p_x", " p_y", " p_z", " M",)]
        out_lines.append(line)
        running_sum = LorentzVector()

        for i in sorted(self.keys()):
            mom = LorentzVector(self[i])
            mom_list = [mom[0], mom[1], mom[2], mom[3], math.sqrt(abs(mom.square()))]
            if i <= n_initial:
                running_sum += mom
            else:
                running_sum -= mom
            ###out_lines.append(template % tuple(['%d' % i] + [special_float_format(el) for el in (list(mom) + [math.sqrt(abs(mom.square()))])]))
            out_lines.append(
                template
                % tuple(["%d" % i] + [special_float_format(el) for el in mom_list])
            )

        out_lines.append(line)

        running_sum_list = [
            running_sum[0],
            running_sum[1],
            running_sum[2],
            running_sum[3],
        ]
        out_lines.append(
            template
            % tuple(
                ["Sum"] + [special_float_format(el) for el in running_sum_list] + [""]
            )
        )

        return "\n".join(out_lines)

    def boost_to_com(self, initial_leg_numbers):
        """ Boost this kinematic configuration back to its c.o.m. frame given the
        initial leg numbers. This is not meant to be generic and here we *want* to crash
        if we encounter a configuration that is not supposed to ever need boosting in the
        MadNkLO construction.
        """

        if len(initial_leg_numbers) == 2:
            if __debug__:
                sqrts = math.sqrt(
                    (
                        self[initial_leg_numbers[0]] + self[initial_leg_numbers[1]]
                    ).square()
                )
                # Assert initial states along the z axis
                assert abs(self[initial_leg_numbers[0]][1] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[1]][1] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[0]][2] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[1]][2] / sqrts) < 1.0e-9
            # Now send the self back into its c.o.m frame, if necessary
            initial_momenta_summed = (
                self[initial_leg_numbers[0]] + self[initial_leg_numbers[1]]
            )
            sqrts = math.sqrt((initial_momenta_summed).square())
            if abs(initial_momenta_summed[3] / sqrts) > 1.0e-9:
                boost_vector = (initial_momenta_summed).boostVector()
                for vec in self.values():
                    vec.boost(-boost_vector)
            if __debug__:
                assert (
                    abs(
                        (self[initial_leg_numbers[0]] + self[initial_leg_numbers[1]])[3]
                        / sqrts
                    )
                    <= 1.0e-9
                )
        elif len(initial_leg_numbers) == 1:
            if __debug__:
                sqrts = math.sqrt(self[initial_leg_numbers[0]].square())
                assert abs(self[initial_leg_numbers[0]][1] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[0]][2] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[0]][3] / sqrts) < 1.0e-9
        else:
            raise InvalidOperation(
                "MadNkLO only supports processes with one or two initial states."
            )

    def get_copy(self):
        """Return a copy that can be freely modified
        without changing the current instance.
        """

        return type(self)((i, LorentzVector(k)) for i, k in self.items())


# =========================================================================================
# LorentzVectorList
# =========================================================================================


class LorentzVectorList(list):
    """A simple class wrapping lists that store Lorentz vectors."""

    def __str__(self, n_initial=2):
        """Nice printout of the momenta."""

        return LorentzVectorDict((i + 1, v) for i, v in enumerate(self)).__str__(
            n_initial=n_initial
        )

    def to_list(self):
        """Return list copy of self."""

        return LorentzVectorList(self)

    def to_array(self):
        """Return a copy as an immutable list of arrays"""

        return list([v.asarray() for v in self])

    def to_tuple(self):
        """Return a copy of this LorentzVectorList as an immutable tuple."""

        return tuple(tuple(v) for v in self)

    def to_dict(self):
        """Return a copy of this LorentzVectorList as a LorentzVectorDict."""

        return LorentzVectorDict((i + 1, v) for i, v in enumerate(self))

    def boost_from_com_to_lab_frame(self, x1, x2, ebeam1, ebeam2):
        """ Boost this kinematic configuration from the center of mass frame to the lab frame
        given specified Bjorken x's x1 and x2.
        This function needs to be cleaned up and built in a smarter way as the boost vector can be written
        down explicitly as a function of x1, x2 and the beam energies.
        """

        if x1 is None:
            x1 = 1.0
        if x2 is None:
            x2 = 1.0

        target_initial_momenta = []
        for i, (x, ebeam) in enumerate(zip([x1, x2], [ebeam1, ebeam2])):
            target_initial_momenta.append(
                LorentzVector(
                    [x * ebeam, 0.0, 0.0, math.copysign(x * ebeam, self[i][3])]
                )
            )
        target_summed = sum(target_initial_momenta)
        source_summed = LorentzVector(
            [2.0 * math.sqrt(x1 * x2 * ebeam1 * ebeam2), 0.0, 0.0, 0.0]
        )

        # We want to send the source to the target
        for vec in self:
            vec.boost_from_to(source_summed, target_summed)
            # boost_vec = LorentzVector.boost_vector_from_to(source_summed, target_summed)
            # import madgraph.various.misc as misc
            # misc.sprint(boost_vec)
            # vec.boost(boost_vec)

    def boost_to_com(self, initial_leg_numbers):
        """ Boost this kinematic configuration back to its c.o.m. frame given the
        initial leg numbers. This is not meant to be generic and here we *want* to crash
        if we encounter a configuration that is not supposed to ever need boosting in the
        MadNkLO construction.
        """
        # Given that this is a list, we must subtract one to the indices given
        initial_leg_numbers = tuple(n - 1 for n in initial_leg_numbers)
        if len(initial_leg_numbers) == 2:
            if __debug__:
                sqrts = math.sqrt(
                    (
                        self[initial_leg_numbers[0]] + self[initial_leg_numbers[1]]
                    ).square()
                )
                # Assert initial states along the z axis
                assert abs(self[initial_leg_numbers[0]][1] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[1]][1] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[0]][2] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[1]][2] / sqrts) < 1.0e-9
            # Now send the self back into its c.o.m frame, if necessary
            initial_momenta_summed = (
                self[initial_leg_numbers[0]] + self[initial_leg_numbers[1]]
            )
            sqrts = math.sqrt((initial_momenta_summed).square())
            if abs(initial_momenta_summed[3] / sqrts) > 1.0e-9:
                boost_vector = (initial_momenta_summed).boostVector()
                for vec in self:
                    vec.boost(-boost_vector)
            if __debug__:
                assert (
                    abs(
                        (self[initial_leg_numbers[0]] + self[initial_leg_numbers[1]])[3]
                        / sqrts
                    )
                    <= 1.0e-9
                )
        elif len(initial_leg_numbers) == 1:
            if __debug__:
                sqrts = math.sqrt(self[initial_leg_numbers[0]].square())
                assert abs(self[initial_leg_numbers[0]][1] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[0]][2] / sqrts) < 1.0e-9
                assert abs(self[initial_leg_numbers[0]][3] / sqrts) < 1.0e-9
        else:
            raise InvalidOperation(
                "MadNkLO only supports processes with one or two initial states."
            )

    def get_copy(self):
        """Return a copy that can be freely modified
        without changing the current instance.
        """

        return type(self)([LorentzVector(p) for p in self])
