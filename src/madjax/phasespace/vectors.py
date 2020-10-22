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
import jax.numpy as jnp
from jax.ops import index_update


logger = logging.getLogger("madgraph.PhaseSpaceGenerator")


class _Vector3(object):
    def __init__(self, vec):
        self.vector = jnp.asarray(vec)

    def __getitem__(self, index):
        return self.vector[index]

    def __truediv__(self, scalar):
        return _Vector3(self.vector / scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __mul__(self, scalar):
        return _Vector3(self.vector * scalar)

    def __mull__(self, scalar):
        return _Vector3(self.vector * scalar)

    def square(self):
        return self.vector.dot(self.vector)

    def asarray(self):
        return self.vector


class _Vector(object):
    def __init__(self, vec):
        self.vector = jnp.asarray(vec)

    def __mul__(self, scalar):
        return _Vector(self.vector * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __getitem__(self, index):
        return self.vector[index]

    def __setitem__(self, index, value):
        self.vector = index_update(self.vector, index, value)

    def __sub__(self, other):
        return _Vector(self.vector - other.vector)

    def __add__(self, other):
        return _Vector(self.vector + other.vector)

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

        return _Vector3(self[1:])

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

        return self.almost_zero(self.square() / jnp.dot(self, self))

    def rho2(self):
        """Compute the radius squared."""
        return self.space().square()

    def rho(self):
        """Compute the radius."""

        return jnp.sqrt(self.space().square())

    def space_direction(self):
        """Compute the corresponding unit vector in ordinary space."""
        return self.space() / self.rho()

    def phi(self):
        return jnp.arctan2(self[2], self[1])

    def boost(self, boost_vector, gamma=-1.0):
        """Transport self into the rest frame of the boost_vector in argument.
        This means that the following command, for any vector p=(E, px, py, pz)
            p.boost(-p.boostVector())
        transforms p to (M,0,0,0).
        """

        b2 = boost_vector.square()

        if gamma < 0.0:
            gamma = 1.0 / jnp.sqrt(1.0 - b2)

        bp = self.space().vector.dot(boost_vector.vector)
        gamma2 = jnp.where(b2 > 0, (gamma - 1.0) / b2, 0.0)
        factor = gamma2 * bp + gamma * self[0]
        # self.space().vector += factor*boost_vector.vector
        self[1:] += factor * boost_vector.vector
        self[0] = gamma * (self[0] + bp)
        return self

    def boostVector(self):
        if self == LorentzVector():
            return Vector([0.0] * 3)
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
        return _Vector(vec)


class LorentzVector(Vector):
    def __new__(cls, *args, **opts):
        if len(args) == 0:
            return super(LorentzVector, cls).__new__(cls, [0.0, 0.0, 0.0, 0.0], **opts)
        return super(LorentzVector, cls).__new__(cls, *args, **opts)
