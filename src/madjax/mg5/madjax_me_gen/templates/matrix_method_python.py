class Matrix_%(process_string)s(object):

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        self.jamp = []

    def get_external_masses(self, model):

        return %(external_masses)s

    def smatrix(self,p, model):
        #  
        %(info_lines)s
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        %(process_lines)s
        #  
        # Clean additional output
        #
        self.clean()
        #  
        # CONSTANTS
        #  
        nexternal = %(nexternal)d
        ndiags = %(ndiags)d
        ncomb = %(ncomb)d
        #  
        # LOCAL VARIABLES 
        #  
        %(helicity_lines)s
        %(den_factor_line)s
        # ----------
        # BEGIN CODE
        # ----------
        self.amp2 = [0.] * ndiags
        self.helEvals = []
        ans = 0.

        # ----------
        # OLD CODE
        # ----------
        #for hel in helicities:
        #    t = self.matrix(p, hel, model)
        #    ans = ans + t
        #    self.helEvals.append([hel, t.real / denominator ])

        t = self.vmap_matrix( p, np.array(helicities), model )
        ans = np.sum(t)
        self.helEvals.append( (helicities, t.real / denominator) )
        
        ans = ans / denominator
        return ans.real
    
    def vmap_matrix(self, p, hel_batch, model):
        return vmap(self.matrix, in_axes=(None,0,None), out_axes=0)(p, hel_batch, model)

    def matrix(self, p, hel, model):
        #  
        %(info_lines)s
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        %(process_lines)s
        #  
        #  
        # Process parameters
        #  
        ngraphs = %(ngraphs)d
        nexternal = %(nexternal)d
        nwavefuncs = %(nwavefuncs)d
        ncolor = %(ncolor)d
        ZERO = 0.
        #  
        # Color matrix
        #  
        %(color_matrix_lines)s
        #
        # Model parameters
        #
        %(model_parameters)s
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        %(helas_calls)s

        jamp = [None] * ncolor

        %(jamp_lines)s

        %(amp2_lines)s

        # ----------
        # OLD CODE
        # ----------
        #matrix = 0.
        #for i in range(ncolor):
        #    ztemp = 0
        #    for j in range(ncolor):
        #        ztemp = ztemp + cf[i][j]*jamp[j]
        #    matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   
        self.jamp.append(jamp)

        matrix = np.sum( np.dot(np.array(cf), np.array(jamp)) * np.array(jamp).conjugate() / np.array(denom) )

        return matrix
