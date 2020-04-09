
import logging
import math
from .vectors import LorentzVector

from functools import partial
from jax import jit, lax
import jax.numpy as jnp

from jax.config import config
config.update('jax_enable_x64', True)

class InvalidCmd(RuntimeError):
    pass

class PhaseSpaceGeneratorError(RuntimeError):
    pass

logger = logging.getLogger('madjax.PhaseSpaceGenerator')

class Dimension(object):
    """ A dimension object specifying a specific integration dimension."""
    
    def __init__(self, name, folded=False):
        self.name   = name
        self.folded = folded
    
    def length(self):
        raise NotImplementedError()
    
    def random_sample(self):        
        raise NotImplementedError()

class DiscreteDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""
    
    def __init__(self, name, values, **opts):
        try:
            self.normalized = opts.pop('normalized')
        except:
            self.normalized = False
        super(DiscreteDimension, self).__init__(name, **opts)
        assert(isinstance(values, list))
        self.values = values
    
    def length(self):
        if self.normalized:
            return 1.0/float(len(self.values))
        else:
            return 1.0
    
    def random_sample(self):
        return jnp.int64(random.choice(self.values))
        
class ContinuousDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""
    
    def __init__(self, name, lower_bound=0.0, upper_bound=1.0, **opts):
        super(ContinuousDimension, self).__init__(name, **opts)
        assert(upper_bound>lower_bound)
        self.lower_bound  = lower_bound
        self.upper_bound  = upper_bound 

    def length(self):
        return (self.upper_bound-self.lower_bound)

    def random_sample(self):
        return jnp.float64(self.lower_bound+random.random()*(self.upper_bound-self.lower_bound))

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
        assert(isinstance(arg, Dimension))
        super(DimensionList, self).append(arg, **opts)
        
    def get_discrete_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, DiscreteDimension))
    
    def get_continuous_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, ContinuousDimension))
    
    def random_sample(self):
        return jnp.array([d.random_sample() for d in self])


#=========================================================================================
# Phase space generation
#=========================================================================================


#================================================
# From Virtual Phase space generation
#================================================
def generate_phase_space_inputs( initial_masses, final_masses,
                                     beam_Es, beam_types=(1,1),
                                     is_beam_factorization_active=(False, False),
                                     correlated_beam_convolution = False ):

    _inputs = {}
    _inputs["initial_masses"]  = initial_masses
    _inputs["masses"]          = final_masses
    _inputs["n_initial"]       = len(initial_masses)
    _inputs["n_final"]         = len(final_masses)
    _inputs["beam_Es"]         = beam_Es
    _inputs["collider_energy"] = sum(beam_Es)
    _inputs["beam_types"]      = beam_types
    _inputs["is_beam_factorization_active"] = is_beam_factorization_active
    _inputs["correlated_beam_convolution"]  = correlated_beam_convolution
    
    # Sanity check
    if _inputs["correlated_beam_convolution"] and _inputs["is_beam_factorization_active"] != (True, True):
        raise PhaseSpaceGeneratorError(
            'The beam convolution cannot be set to be correlated if it is one-sided only')
    
    dimensions                 = _get_dimensions(_inputs["n_initial"], _inputs["n_final"],
                                                     _inputs["beam_types"],
                                                     _inputs["correlated_beam_convolution"],
                                                     _inputs["is_beam_factorization_active"] )
    
    _inputs["dim_ordered_names"]    = [d.name for d in dimensions]
    _inputs["dim_name_to_position"] = dict((d.name,i) for i, d in enumerate(dimensions))
    _inputs["position_to_dim_name"] = dict((v,k) for (k,v) in _inputs["dim_name_to_position"].items())

    return _inputs

def boost_to_lab_frame(PS_point, n_initial, xb_1, xb_2):
    """Boost a phase-space point from the COM-frame to the lab frame, given Bjorken x's."""
    
    if n_initial == 2 and (xb_1!=1. or xb_2!=1.):
        ref_lab = (PS_point[0]*xb_1 + PS_point[1]*xb_2)
        if ref_lab.rho2() != 0.:
            lab_boost = ref_lab.boostVector()
            for p in PS_point:
                p.boost(-lab_boost)
    return


def boost_to_COM_frame(PS_point, n_initial):
    """Boost a phase-space point from the lab frame to the COM frame"""
        
    if n_initial == 2:
        ref_com = (PS_point[0] + PS_point[1])
        if ref_com.rho2() != 0.:
            com_boost = ref_com.boostVector()
            for p in PS_point:
                p.boost(-com_boost)

    return

    

def nDimPhaseSpace(n_final):
    """Return the number of random numbers required to produce
    a given multiplicity final state."""

    if n_final == 1:
        return 0
    return 3*n_final - 4



def _get_dimensions(n_initial, n_final, beam_types, correlated_beam_convolution, is_beam_factorization_active):
    """Generate a list of dimensions for this integrand."""
    
    dims = DimensionList()

    # Add the PDF dimensions if necessary
    if beam_types[0]==beam_types[1]==1:
        dims.append(ContinuousDimension('ycms',lower_bound=0.0, upper_bound=1.0))
        # The 2>1 topology requires a special treatment
        if not (n_initial==2 and n_final==1):
            dims.append(ContinuousDimension('tau',lower_bound=0.0, upper_bound=1.0)) 

    # Add xi beam factorization convolution factors if necessary
    if correlated_beam_convolution:
        # A single convolution factor xi that applies to both beams is needed in this case
        dims.append(ContinuousDimension('xi',lower_bound=0.0, upper_bound=1.0))
    else:
        if is_beam_factorization_active[0]:
            dims.append(ContinuousDimension('xi1',lower_bound=0.0, upper_bound=1.0))             
        if is_beam_factorization_active[1]:
            dims.append(ContinuousDimension('xi2',lower_bound=0.0, upper_bound=1.0))

    # Add the phase-space dimensions
    dims.extend([ ContinuousDimension('x_%d'%i,lower_bound=0.0, upper_bound=1.0)
                      for i in range(1, nDimPhaseSpace(n_final)+1) ])
        
    return dims



#================================================
# From Flat Phase space generation
# Implementation following S. Platzer, arxiv:1308.2922
#================================================

def epsilon_border():
    # This parameter defines a thin layer around the boundary of the unit hypercube
    # of the random variables generating the phase-space,
    # so as to avoid extrema which are an issue in most PS generators.
    return 1e-10

def absolute_Ecm_min():
    # The lowest value that the center of mass energy can take.
    # We take here 1 GeV, as anyway below this non-perturbative effects dominate
    # and factorization does not make sense anymor
    return 1.0

def flatWeights():
    # For reference here we put the flat weights that Simon uses in his
    # Herwig implementation. I will remove them once I will have understood
    # why they don't match the physical PS volume.
    # So these are not used for now, and get_flatWeights() is used instead
    return { 2 :  0.039788735772973833942,
                 3 :  0.00012598255637968550463,
                 4 :  1.3296564302788840628e-7,
                 5 :  7.0167897579949011130e-11,
                 6 :  2.2217170114046130768e-14 }


def get_dimensions(PS_inputs):
    #Make sure the collider setup is supported.

    beam_types                   = PS_inputs["beam_types"]
    beam_Es                      = PS_inputs["beam_Es"]
    n_initial                    = PS_inputs["n_initial"]
    n_final                      = PS_inputs["n_final"]
    correlated_beam_convolution  = PS_inputs["correlated_beam_convolution"],
    is_beam_factorization_active = PS_inputs["is_beam_factorization_active"]

    # Check if the beam configuration is supported
    if (not abs(beam_types[0])==abs(beam_types[1])==1) and \
      (not beam_types[0]==beam_types[1]==0):
        raise InvalidCmd(
            "This basic generator does not support the specified collider configuration")
        
    if beam_Es[0]!=beam_Es[1]:
        raise InvalidCmd(
            "This basic generator only supports colliders with incoming beams equally energetic.")

    return _get_dimensions(n_initial, n_final,
                               beam_types,
                               correlated_beam_convolution,
                               is_beam_factorization_active)



def get_flatWeights(E_cm, n, mass=None):
    # Return the phase-space volume for a n massless final states.
    # Vol(E_cm, n) = (pi/2)^(n-1) *  (E_cm^2)^(n-2) / ((n-1)!*(n-2)!)
    
    if n==1:
        # The jacobian from \delta(s_hat - m_final**2) present in 2->1 convolution
        # must typically be accounted for in the MC integration framework since we
        # don't have access to that here, so we just return 1.
        return 1.

    #out_ = jnp.where(n==1, 1.0,
    #                    jnp.power((math.pi/2.0),n-1)*(jnp.power((E_cm**2),n-2)/(math.factorial(n-1)*math.factorial(n-2)))
    #                    )
    #return out_

    return jnp.power((math.pi/2.0),n-1)*\
      (jnp.power((E_cm**2),n-2)/(math.factorial(n-1)*math.factorial(n-2)))


def bisect(v,n):
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
            low = jnp.where(update_upper, low, midpoint)
            high = jnp.where(update_upper, midpoint, high)
            return (low, high)

        solution, _ = lax.while_loop(cond, body, (low, high))
        return solution
    tangent_solve=scalar_solve
    def f(u):
        return (u**(n+1)) * (n+2.-(n+1.)*u)-v
    return lax.custom_root(f, 0.5, binary_search, tangent_solve)
    


def rho(M, N, m):
    #Returns sqrt((sqr(M)-sqr(N+m))*(sqr(M)-sqr(N-m)))/(8.*sqr(M))

    #Msqr = M**2
    #return ((Msqr-(N+m)**2) * (Msqr-(N-m)**2) )**0.5 / (8.*Msqr)

    Msqr = jnp.power(M,2)
    return jnp.power( (Msqr - jnp.power((N+m),2)) *
                         (Msqr-jnp.power((N-m),2)), 0.5) / (8.*Msqr)



def setInitialStateMomenta(PS_inputs, output_momenta, E_cm):
    #Generate the initial state momenta."""

    n_initial      = PS_inputs["n_initial"]
    initial_masses = PS_inputs["initial_masses"]
    
    if n_initial not in [1,2]:
        raise InvalidCmd("This PS generator only supports 1 or 2 initial states")

    if n_initial == 1:
        if self.initial_masses[0]==0.:
            raise PhaseSpaceGeneratorError("Cannot generate the decay phase-space of a massless particle.")
        
        if E_cm != initial_masses[0]:
            raise PhaseSpaceGeneratorError("Can only generate the decay phase-space of a particle at rest.")

    if n_initial == 1:
        output_momenta[0] = LorentzVector([initial_masses[0] , 0., 0., 0.])
        return

    elif n_initial == 2:
        if initial_masses[0] == 0. or initial_masses[1] == 0.:
            output_momenta[0] = LorentzVector([E_cm/2.0 , 0., 0., 1.0*E_cm/2.0])
            output_momenta[1] = LorentzVector([E_cm/2.0 , 0., 0., (-1.0)*E_cm/2.0])
        else:
            M1sq = initial_masses[0]**2
            M2sq = initial_masses[1]**2
            E1 = (E_cm**2+M1sq-M2sq)/ E_cm
            E2 = (E_cm**2-M1sq+M2sq)/ E_cm
            Z = jnp.sqrt(E_cm**4 - 2*E_cm**2*M1sq - 2*E_cm**2*M2sq + M1sq**2 - 2*M1sq*M2sq + M2sq**2) / E_cm
            output_momenta[0] = LorentzVector([E1/2.0 , 0., 0., 1.0*Z/2.0])
            output_momenta[1] = LorentzVector([E2/2.0 , 0., 0., (-1.0)*Z/2.0])
    return


def get_PS_point(PS_inputs, random_variables):
    #Generate a complete PS point, including Bjorken x's,
    #dictating a specific choice of incoming particle's momenta.

    dim_name_to_position         = PS_inputs["dim_name_to_position"]
    correlated_beam_convolution  = PS_inputs["correlated_beam_convolution"]
    is_beam_factorization_active = PS_inputs["is_beam_factorization_active"]

    collider_energy              = PS_inputs["collider_energy"]
    beam_types                   = PS_inputs["beam_types"]
    masses                       = PS_inputs["masses"]
    n_initial                    = PS_inputs["n_initial"]
    n_final                      = PS_inputs["n_final"]


    dimensions = _get_dimensions(n_initial, n_final,
                                     beam_types,
                                     correlated_beam_convolution,
                                     is_beam_factorization_active )
    

    # if random_variables are not defined, than just throw a completely random point
    if random_variables is None:
            random_variables = dimensions.random_sample()
        
    # Check the sensitivity of te inputs from the integrator
    # Lukas: not part of computation ignore for now
    # if jnp.any(jnp.isnan(random_variables)):
    #     logger.warning('Some input variables from the integrator are malformed: %s'%
    #         ( ', '.join( '%s=%s'%( name, random_variables[pos]) for name, pos in 
    #                                              self.dim_name_to_position.items() ) ))
    #     logger.warning('The PS generator will yield None, triggering the point to be skipped.')
    #     return None, 0.0, (0., 0.), (0., 0.)
    
    # Phase-space point weight to return
    wgt = 1.0
    
    #if any(math.isnan(r) for r in random_variables):
    #    misc.sprint(random_variables)
    
    # Avoid extrema since the phase-space generation algorithm doesn't like it
    random_variables = [min(max(rv, epsilon_border()), 1.-epsilon_border()) for rv in random_variables]

    # Assign variables to their meaning.
    if 'ycms' in dim_name_to_position:
        PDF_ycm = random_variables[dim_name_to_position['ycms']]
    else:
        PDF_ycm = None
    if 'tau' in dim_name_to_position:
        PDF_tau = random_variables[dim_name_to_position['tau']]
    else:
        PDF_tau = None
        
    PS_random_variables  = [rv for i, rv in enumerate(random_variables) if position_to_dim_name[i].startswith('x_') ]

    # Also generate the ISR collinear factorization convolutoin variables xi<i> if
    # necessary. In order for the + distributions of the PDF counterterms and integrated
    # collinear ISR counterterms to hit the PDF only (and not the matrix elements or
    # observables functions), a change of variable is necessary: xb_1' = xb_1 * xi1
    if correlated_beam_convolution:
        # Both xi1 and xi2 must be set equal then
        xi1 = random_variables[dim_name_to_position['xi']]
        xi2 = random_variables[dim_name_to_position['xi']]
    else:
        if is_beam_factorization_active[0]:
            xi1 = random_variables[dim_name_to_position['xi1']]
        else:
            xi1 = None
        if is_beam_factorization_active[1]:
            xi2 = random_variables[dim_name_to_position['xi2']]
        else:
            xi2 = None

    # Now take care of the Phase-space generation:
    # Set some defaults for the variables to be set further
    xb_1 = 1.
    xb_2 = 1.
    E_cm = collider_energy
        
    # We generate the PDF from two variables \tau = x1*x2 and ycm = 1/2 * log(x1/x2), so that:
    #  x_1 = sqrt(tau) * exp(+ycm)
    #  x_2 = sqrt(tau) * exp(-ycm)
    # The jacobian of this transformation is 1.
    if abs(beam_types[0])==abs(beam_types[1])==1:
            
        tot_final_state_masses = sum(masses)
        if tot_final_state_masses > collider_energy:
            raise PhaseSpaceGeneratorError("Collider energy is not large enough, there is no phase-space left.")
            
        # Keep a hard cut at 1 GeV, which is the default for absolute_Ecm_min
        tau_min = (max(tot_final_state_masses, absolute_Ecm_min())/collider_energy)**2
        tau_max = 1.0
        
        if n_initial == 2 and n_final == 1:
            # Here tau is fixed by the \delta(xb_1*xb_2*s - m_h**2) which sets tau to 
            PDF_tau = tau_min
            # Account for the \delta(xb_1*xb_2*s - m_h**2) and corresponding y_cm matching to unit volume
            wgt *= (1./collider_energy**2)
        else:
            # Rescale tau appropriately
            PDF_tau = tau_min+(tau_max-tau_min)*PDF_tau
            # Including the corresponding Jacobian
            wgt *= (tau_max-tau_min)
            
        # And we can now rescale ycm appropriately
        ycm_min = 0.5 * jnp.log(PDF_tau)
        ycm_max = -ycm_min
        PDF_ycm = ycm_min + (ycm_max - ycm_min)*PDF_ycm            
        # and account for the corresponding Jacobian
        wgt *= (ycm_max - ycm_min)
        
        xb_1 = jnp.sqrt(PDF_tau) * jnp.exp(PDF_ycm)
        xb_2 = jnp.sqrt(PDF_tau) * jnp.exp(-PDF_ycm)
        # /!\ The mass of initial state momenta is neglected here.
        E_cm = jnp.sqrt(xb_1*xb_2)*collider_energy
        
    elif beam_types[0]==beam_types[1]==0:
        xb_1 = 1.
        xb_2 = 1.
        E_cm = collider_energy
    else:
        raise InvalidCmd("This basic PS generator does not yet support collider mode (%d,%d)."%beam_types)

    # Now generate a PS point
    PS_point, PS_weight = generateKinematics(PS_inputs, E_cm, PS_random_variables)
        
    # Apply the phase-space weight
    wgt *= PS_weight
        
    return PS_point, wgt, (xb_1, xi1) , (xb_2, xi2)



def generateKinematics(PS_inputs, E_cm, random_variables):
    #Generate a self.n_initial -> self.n_final phase-space point
    #using the random variables passed in argument.

    masses    = PS_inputs["masses"]
    n_initial = PS_inputs["n_initial"]
    n_final   = PS_inputs["n_final"]

    # Make sure the right number of random variables are passed
    
    try:
        assert len(random_variables) == nDimPhaseSpace((n_final))
    except:
        raise RuntimeError('need {} random variables'.format(nDimPhaseSpace(n_final)))

    # Make sure that none of the random_variables is NaN.
    # Lukas: not part of computation ignore for now
    # if jnp.any(jnp.isnan(random_variables)):
    #     raise PhaseSpaceGeneratorError("Some of the random variables passed "+
    #       "to the phase-space generator are NaN: %s"%str(random_variables))

    # The distribution weight of the generate PS point
    weight = 1.
        
    output_momenta = []

    mass = masses[0]
    if n_final == 1:
        if n_initial == 1:
            raise InvalidCmd("1 > 1 phase-space generation not supported.")
        if mass/E_cm < 1.e-7 or ((E_cm-mass)/mass) > 1.e-7:
            raise PhaseSpaceGeneratorError("1 > 2 phase-space generation needs a final state mass equal to E_c.o.m.")
            
        output_momenta.append(LorentzVector([mass/2., 0., 0., 1.0*mass/2.]))
        output_momenta.append(LorentzVector([mass/2., 0., 0., (-1.0)*mass/2.]))
        output_momenta.append(LorentzVector([mass   , 0., 0.,       0.]))
        weight = get_flatWeights(E_cm, 1)
        return output_momenta, weight
  
    M    = [ 0. ]*(n_final-1)
    M[0] = E_cm
    
    weight *= generateIntermediatesMassive(n_final, masses, M, E_cm, random_variables)
    M.append(masses[-1])
    
    
    Q     = LorentzVector([M[0], 0., 0., 0.])
    nextQ = LorentzVector()
    
    for i in range(n_initial+n_final-1):
            
        if i < n_initial:
            output_momenta.append(LorentzVector())
            continue

        q = 4.*M[i-n_initial]*rho( M[i-n_initial],M[i-n_initial+1],masses[i-n_initial] )
        cos_theta = 2.*random_variables[n_final-2+2*(i-n_initial)]-1.
        sin_theta = jnp.sqrt(1.-cos_theta**2)
        phi = 2.*math.pi*random_variables[n_final-1+2*(i-n_initial)]
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sqrt(1.-cos_phi**2)

        sin_phi = jnp.where(phi > math.pi, (-1.0)*sin_phi,sin_phi)
        # if (phi > math.pi):
        #     sin_phi = -sin_phi
            
        p = LorentzVector([0., q*sin_theta*cos_phi, q*sin_theta*sin_phi, q*cos_theta])
        p.set_square(masses[i-n_initial]**2)
        p.boost(Q.boostVector())
        p.set_square(masses[i-n_initial]**2)
        output_momenta.append(p)
        
        nextQ = Q - p
        nextQ.set_square(M[i-n_initial+1]**2)
        Q = nextQ
       
    output_momenta.append(Q)
    
    setInitialStateMomenta(PS_inputs, output_momenta, E_cm)

    return output_momenta, weight



def generateIntermediatesMassless(n_final, M, E_cm, random_variables):
    #Generate intermediate masses for a massless final state."""
    
    for i in range(2, n_final):
        u = bisect(random_variables[i-2], n_final-1-i)
        M[i-1] = jnp.sqrt(u*(M[i-2]**2))

    return get_flatWeights(E_cm, n_final)


   
def generateIntermediatesMassive(n_final, masses, M, E_cm, random_variables):
    #Generate intermediate masses for a massive final state."""

    K = list(M)
    K[0] -= sum(masses)
    
    weight = generateIntermediatesMassless(n_final, K, E_cm, random_variables)
    
    del M[:]
    M.extend(K)
    
    for i in range(1, n_final):
        for k in range(i, n_final+1):
            M[i-1] +=  masses[k-1]
        
    weight *= 8.*rho( M[n_final-2], masses[n_final-1], masses[n_final-2] )
    

    for i in range(2, n_final):
        weight *= (rho(M[i-2],M[i-1], masses[i-2]) / rho(K[i-2],K[i-1],0.)) * (M[i-1]/K[i-1])

    
    weight *= jnp.power(K[0]/M[0], 2*n_final-4)
    

    return weight


def invertKinematics(PS_inputs, E_cm, momenta):
    #Returns the random variables that yields the specified momenta configuration."""

    n_initial = PS_inputs["n_initial"]
    n_final   = PS_inputs["n_final"]
    masses    = PS_inputs["masses"]

    # Make sure the right number of momenta are passed
    assert (len(momenta) == (n_initial + n_final) )
    moms = momenta.copy()

    # The weight of the corresponding PS point
    weight = 1.

    if n_final == 1:
        if n_initial == 1:
            raise PhaseSpaceGeneratorError("1 > 1 phase-space generation not supported.")
        return [], get_flatWeights(E_cm,1) 

    # The random variables that would yield this PS point.
    random_variables = [-1.0]*nDimPhaseSpace(n_final)
        
    M    = [0., ]*(n_final-1)
    M[0] = E_cm

    Q     = [LorentzVector(), ]*(n_final-1)
    Q[0]  = LorentzVector([M[0],0.,0.,0.])

    for i in range(2,n_final):
        for k in range(i, n_final+1):
            Q[i-1] = Q[i-1] + moms[k+n_initial-1]
        M[i-1] = abs(Q[i-1].square()) ** 0.5

    weight = invertIntermediatesMassive(n_final, masses, M, E_cm, random_variables)

    for i in range(n_initial, n_final+1):
        # BALDY another copy? moms not used afterwards
        p = LorentzVector(moms[i])
        # Take the opposite boost vector
        boost_vec = -1.0*Q[i-n_initial].boostVector()
        p.boost(boost_vec)
        random_variables[n_final-2+2*(i-n_initial)] = (p.cosTheta()+1.)/2.
        phi = p.phi()
        if (phi < 0.):
            phi += 2.*math.pi
        random_variables[n_final-1+2*(i-n_initial)] = phi / (2.*math.pi)
        
    return random_variables, weight



def invertIntermediatesMassive(n_final, masses, M, E_cm, random_variables):
    #Invert intermediate masses for a massive final state."""
    
    K = list(M)
    for i in range(1, n_final):
        K[i-1] -= sum(masses[i-1:])
        
    weight = invertIntermediatesMassless(n_final, K, E_cm, random_variables)
    weight *= 8.*rho(M[n_final-2], masses[n_final-1], masses[n_final-2])
    
    for i in range(2, n_final):
        weight *= (rho(M[i-2],M[i-1], masses[i-2])/ rho(K[i-2],K[i-1],0.)) * (M[i-1]/K[i-1])
        
    weight *= jnp.power(K[0]/M[0], 2*n_final-4)

    return weight


def invertIntermediatesMassless(n_final, K, E_cm, random_variables):
    #Invert intermediate masses for a massless final state."""

    for i in range(2, n_final):
        u = (K[i-1]/K[i-2])**2
        random_variables[i-2] = \
                (n_final+1-i)*jnp.power(u, n_final-i) - \
                (n_final-i)*jnp.power(u, n_final+1-i)
        
    return get_flatWeights(E_cm, n_final)

