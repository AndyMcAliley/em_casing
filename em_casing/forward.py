'''
Integral equation-based forward modeling
'''

import numpy as np
import numpy as np
from empymod import bipole, dipole
from halfspace import form_A, form_b

mu0 = 4e-7*np.pi

def greens(zp,yp,xp,z,y,x,src_cmp,rec_cmp,**epy_args):
