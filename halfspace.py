'''
A module for modeling electromagnetic fields in a halfspace with a vertical casing

functions:

hankel_J1_140
fii
fij
Aii
Aij
form_A
HED_Ez
HEB_Ez
form_b_analytic
form_b

'''

import numpy as np

mu0 = 4e-7*np.pi

def hankel_J1_140(function, r):
    '''
    Compute Hankel transform of order 1:
    f(r) = integral 0 -> inf (function(lamda) * J1(r*lamda) * d lamda)
    Follows Guptasarma and Singh (1997)
    '''
    a = -7.91001919000
    s = 0.0879671439570
    W = [-6.76671159511e-14,
         3.39808396836e-13,
         -7.43411889153e-13,
         8.93613024469e-13,
         -5.47341591896e-13,
         -5.84920181906e-14,
         5.20780672883e-13,
         -6.92656254606e-13,
         6.88908045074e-13,
         -6.39910528298e-13,
         5.82098912530e-13,
         -4.84912700478e-13,
         3.54684337858e-13,
         -2.10855291368e-13,
         1.00452749275e-13,
         5.58449957721e-15,
         -5.67206735175e-14,
         1.09107856853e-13,
         -6.04067500756e-14,
         8.84512134731e-14,
         2.22321981827e-14,
         8.38072239207e-14,
         1.23647835900e-13,
         1.44351787234e-13,
         2.94276480713e-13,
         3.39965995918e-13,
         6.17024672340e-13,
         8.25310217692e-13,
         1.32560792613e-12,
         1.90949961267e-12,
         2.93458179767e-12,
         4.33454210095e-12,
         6.55863288798e-12,
         9.78324910827e-12,
         1.47126365223e-11,
         2.20240108708e-11,
         3.30577485691e-11,
         4.95377381480e-11,
         7.43047574433e-11,
         1.11400535181e-10,
         1.67052734516e-10,
         2.50470107577e-10,
         3.75597211630e-10,
         5.63165204681e-10,
         8.44458166896e-10,
         1.26621795331e-09,
         1.89866561359e-09,
         2.84693620927e-09,
         4.26886170263e-09,
         6.40104325574e-09,
         9.59798498616e-09,
         1.43918931885e-08,
         2.15798696769e-08,
         3.23584600810e-08,
         4.85195105813e-08,
         7.27538583183e-08,
         1.09090191748e-07,
         1.63577866557e-07,
         2.45275193920e-07,
         3.67784458730e-07,
         5.51470341585e-07,
         8.26916206192e-07,
         1.23991037294e-06,
         1.85921554669e-06,
         2.78777669034e-06,
         4.18019870272e-06,
         6.26794044911e-06,
         9.39858833064e-06,
         1.40925408889e-05,
         2.11312291505e-05,
         3.16846342900e-05,
         4.75093313246e-05,
         7.12354794719e-05,
         1.06810848460e-04,
         1.60146590551e-04,
         2.40110903628e-04,
         3.59981158972e-04,
         5.39658308918e-04,
         8.08925141201e-04,
         1.21234066243e-03,
         1.81650387595e-03,
         2.72068483151e-03,
         4.07274689463e-03,
         6.09135552241e-03,
         9.09940027636e-03,
         1.35660714813e-02,
         2.01692550906e-02,
         2.98534800308e-02,
         4.39060697220e-02,
         6.39211368217e-02,
         9.16763946228e-02,
         1.28368795114e-01,
         1.73241920046e-01,
         2.19830379079e-01,
         2.51193131178e-01,
         2.32380049895e-01,
         1.17121080205e-01,
         -1.17252913088e-01,
         -3.52148528535e-01,
         -2.71162871370e-01,
         2.91134747110e-01,
         3.17192840623e-01,
         -4.93075681595e-01,
         3.11223091821e-01,
         -1.36044122543e-01,
         5.12141261934e-02,
         -1.90806300761e-02,
         7.57044398633e-03,
         -3.25432753751e-03,
         1.49774676371e-03,
         -7.24569558272e-04,
         3.62792644965e-04,
         -1.85907973641e-04,
         9.67201396593e-05,
         -5.07744171678e-05,
         2.67510121456e-05,
         -1.40667136728e-05,
         7.33363699547e-06,
         -3.75638767050e-06,
         1.86344211280e-06,
         -8.71623576811e-07,
         3.61028200288e-07,
         -1.05847108097e-07,
         -1.51569361490e-08,
         6.67633241420e-08,
         -8.33741579804e-08,
         8.31065906136e-08,
         -7.53457009758e-08,
         6.48057680299e-08,
         -5.37558016587e-08,
         4.32436265303e-08,
         -3.37262648712e-08,
         2.53558687098e-08,
         -1.81287021528e-08,
         1.20228328586e-08,
         -7.10898040664e-09,
         3.53667004588e-09,
         -1.36030600198e-09,
         3.52544249042e-10,
         -4.53719284366e-11]
    lamda = 10**(a+np.arange(140)*s)/r
    K = function(lamda)
    f = np.dot(K,W)
    return f/r

def fii(lamda,z,dz,frequency,conductivity):
    '''
    function in Hankel transform for Aii elements
    lamda is an array of lamda values
    z is vertical coord of center of ith segment
    dz is length of ith segment
    '''
    # mu0 = 4*np.pi*1e-7
    s = np.sqrt(lamda**2 - 1j*2*np.pi*frequency*mu0*conductivity)
    result = 2*np.exp(-s*dz/2)
    result -= 2
    result -= np.exp(-s*(2*z+dz/2))
    result += np.exp(-s*(2*z-dz/2))
    result *= lamda**2/s**2
    return result

def fij(lamda,zi,zj,dz,frequency,conductivity):
    '''
    function in Hankel transform for Aij elements
    lamda is an array of lamda values
    zi is vertical coord of center of ith segment
    zj is vertical coord of center of jth segment
    dz is length of segments
    '''
    # mu0 = 4*np.pi*1e-7
    s = np.sqrt(lamda**2 - 1j*2*np.pi*frequency*mu0*conductivity)
    result = np.exp(-s*(abs(zi-zj)+dz/2))
    result -= np.exp(-s*(abs(zi-zj)-dz/2))
    result -= np.exp(-s*(zi+zj+dz/2))
    result += np.exp(-s*(zi+zj-dz/2))
    result *= lamda**2/s**2
    return result

def Aii(zi,
        dz,
        frequency=0.125,
        background_conductivity=0.18,
        casing_conductivity=1.0e7,
        outer_radius=0.1095,
        inner_radius=0.1095-0.0134):
    '''
    Form ith diagonal of coefficient matrix to solve for casing current densities
    '''
    funii = lambda x:fii(x,zi,dz,frequency,background_conductivity)
    result = outer_radius*hankel_J1_140(funii,outer_radius)
    result -= inner_radius*hankel_J1_140(funii,inner_radius)
    result = 1/casing_conductivity + result/2/background_conductivity
    return result

def Aij(zi,
        zj,
        dz,
        frequency=0.125,
        background_conductivity=0.18,
        casing_conductivity=1.0e7,
        outer_radius=0.1095,
        inner_radius=0.1095-0.0134):
    '''
    Form i,jth element of coefficient matrix
    '''
    funij = lambda x:fij(x,zi,zj,dz,frequency,background_conductivity)
    result = outer_radius*hankel_J1_140(funij,outer_radius)
    result -= inner_radius*hankel_J1_140(funij,inner_radius)
    result /= 2*background_conductivity
    return result

def form_A(frequency=0.125,
          background_conductivity=0.18,
          casing_conductivity=1.0e7,
          outer_radius=0.1095,
          inner_radius=0.1095-0.0134,
          casing_length=1365,
          num_segments=280,
          **kwargs):
    '''
    Form coefficient matrix to solve for casing current densities
    '''
    dz = casing_length/num_segments
    zs = dz*(np.arange(num_segments)+0.5)
    A = np.ones((num_segments,num_segments))*1j
    diag = np.ones(num_segments)*1j
    #TODO: vectorize or parallelize
    for ii in np.arange(num_segments):
        zi = zs[ii]
        for jj in np.arange(num_segments):
            zj = zs[jj]
            if ii==jj:
                A[ii,jj] = Aii(zi,
                               dz,
                               frequency=frequency,
                               background_conductivity=background_conductivity,
                               casing_conductivity=casing_conductivity,
                               outer_radius=outer_radius,
                               inner_radius=inner_radius
                              )
            else:
                A[ii,jj] = Aij(zi,
                               zj,
                               dz,
                               frequency=frequency,
                               background_conductivity=background_conductivity,
                               casing_conductivity=casing_conductivity,
                               outer_radius=outer_radius,
                               inner_radius=inner_radius
                              )
    return A   

def HED_Ez(x,y,z,xp=0,yp=0,angle=0,moment=1,conductivity=1,frequency=1):
    '''
    z component of electric field due to a horizontal electric dipole
    x,y,z are location of observation, either as scalars or arrays
    xp,yp are location of dipole at surface
    angle is dipole direction in radians from x axis
    Derived from Hohmann and Ward
    Corroborated by Bannister and Dube, 1978, eq. 36
    '''
    k_squared = -1j*2*np.pi*frequency*mu0*conductivity
    r_squared = (x-xp)**2+(y-yp)**2+z**2
    kr_squared = k_squared*r_squared
    ikr = 1j*np.sqrt(kr_squared)
    r = np.sqrt(r_squared)
    # k = np.sqrt(k_squared)

    # distance and depth scaling
    ez = np.exp(-ikr)*z/r**5
    # horizontal distance in the direction of the dipole
    ez *= (x-xp)*np.cos(angle)+(y-yp)*np.sin(angle)
    # unitless term in parenthesis
    ez *= 3+3*ikr-kr_squared
    # scaling factor
    ez *= moment/2/np.pi/conductivity
    return ez

def HEB_Ez(x,y,z,xp1=0,xp2=1,yp1=0,yp2=0,current=1,conductivity=1,frequency=1):
    '''
    z component of electric field due to a horizontal electric bipole
    x,y,z are location of observation, either as scalars or arrays
    xp1,yp1,xp2,yp2 are locations of ends of bipole at surface
    Derived from Hohmann and Ward
    '''
    k_squared = -1j*2*np.pi*frequency*mu0*conductivity
    k = np.sqrt(k_squared)
    r1_squared = (x-xp1)**2+(y-yp1)**2+z**2
    r2_squared = (x-xp2)**2+(y-yp2)**2+z**2
    r1 = np.sqrt(r1_squared)
    r2 = np.sqrt(r2_squared)
    ikr1 = 1j*k*r1
    ikr2 = 1j*k*r2

    # r2
    ez = np.exp(-ikr2)/r2**3*(1+ikr2)
    # r1
    ez -= np.exp(-ikr1)/r1**3*(1+ikr1)
    # scaling factor
    ez *= current*z/2/np.pi/conductivity
    return ez

def form_b_analytic(wire_path_x,
                    wire_path_y,
                    wire_current = 1,
                    frequency=0.125,
                    background_conductivity=0.18,
                    casing_length=1365,
                    num_segments=280,
                    **kwargs):
    '''
    Form the jth element of the RHS vector to solve for casing currents
    
    wire_path: A big honkin' numpy array of x and y positions
        x in column 1, y in column 2
        origin at casing
        These are nodes, so there must be k+1 elements where k is # wire dipoles
    
    NOTE: the analytic solution for Ez only uses the grounding points
    '''
    dz = casing_length/num_segments
    zs = dz*(np.arange(num_segments)+0.5)
    xs = np.zeros(num_segments)
    ys = np.zeros(num_segments)
    return HEB_Ez(xs,ys,zs,
                  xp1=wire_path_x[0],
                  yp1=wire_path_y[0],
                  xp2=wire_path_x[-1],
                  yp2=wire_path_y[-1],
                  current=wire_current,
                  conductivity=background_conductivity,
                  frequency=frequency)

def form_b(*args,**kwargs):
    '''
    Wrapper,
    to be changed to point to different functions when desired
    '''
    return form_b_analytic(*args,**kwargs)

