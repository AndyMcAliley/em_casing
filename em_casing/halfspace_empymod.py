'''
Functions for modeling in a halfspace that require empymod
'''

import numpy as np
from empymod import bipole
from halfspace import form_A, form_b

def wire_e_field_casing_halfspace(tx_path_x,
                                  tx_path_y,
                                  rx_ex_locations,
                                  rx_ey_locations,
                                  frequency,
                                  background_conductivity=0.18,
                                  casing_location=[0,0],
                                  casing_length=1365,
                                  casing_conductivity=1.0e7,
                                  outer_radius=0.1095,
                                  inner_radius=0.1095-0.0134,
                                  num_segments=280,
                                  wire_current=1,
                                  srcpts=1):
    '''
    Compute EM field at rx_locations due to a wire in the presence of a steel casing

    Parameters
    ----------
    tx_path_x : list
        Source coordinates in x (m)

    tx_path_y : list
        Source coordinates in y (m)

    rx_ex_locations, rx_ey_locations : list of lists
        Ex and Ex receiver coordinates (m):
            empymod format
            [x1,x2,y1,y2,z1,z2]
            z1 and z2 can be scalars even when x and y variables are lists

    casing_location : list
        NOT YET IMPLEMENTED: casing assumed to be at 0,0
        casing well head location, as [x,y]

    '''

    # TODO: set well location as origin
    # TODO: compute magnetic field too
    # TODO: allow multiple frequencies

    # discretizations
    # need wire segment lengths
    dlx = np.diff(tx_path_x)
    dly = np.diff(tx_path_y)
    all_segment_lengths = np.sqrt(dlx**2+dly**2)
    # is segment length zero?
    nonzero_length = np.append(all_segment_lengths>0,True)
    # remove 0 length segments
    lx = tx_path_x[nonzero_length]
    ly = tx_path_y[nonzero_length]
    segment_lengths = all_segment_lengths[all_segment_lengths>0]
    # casing discretization
    dz = casing_length/num_segments
    zs = dz*(np.arange(num_segments)+0.5)

    # compute casing currents
    # make an argument dictionary for MoM functions
    casing_args = {'wire_path_x':lx,
                   'wire_path_y':ly,
                   'wire_current':wire_current,
                   'frequency':frequency,
                   'background_conductivity':background_conductivity,
                   'casing_conductivity':casing_conductivity,
                   'casing_length':casing_length,
                   'outer_radius':outer_radius,
                   'inner_radius':inner_radius,
                   'num_segments':num_segments}
    # form A
    A = form_A(**casing_args)
    # form b
    b = form_b(**casing_args)
    # solve for casing currents
    j_casing = np.linalg.solve(A,b)
    casing_area = np.pi*(outer_radius**2-inner_radius**2)
    i_casing = j_casing*casing_area
    casing_moment = i_casing*dz

    # convert inputs to empymod format
    empy_wire_args = {'src':[lx[:-1],lx[1:],ly[:-1],ly[1:],1e-2,1e-2],
                      'rec':rx_ex_locations,
                      'depth':[0],
                      'res':[1e20,1/background_conductivity],
                      'freqtime':frequency,
                      'srcpts':srcpts,
                      'verb':0,
                      'epermH':[0,1],
                      'epermV':[0,1]}
    rx_locations_ex_average = [(rx_ex_locations[0]+rx_ex_locations[1])/2, 
                               (rx_ex_locations[2]+rx_ex_locations[3])/2,
                               (rx_ex_locations[4]+rx_ex_locations[5])/2]
    rx_locations_ey_average = [(rx_ey_locations[0]+rx_ey_locations[1])/2, 
                               (rx_ey_locations[2]+rx_ey_locations[3])/2,
                               (rx_ey_locations[4]+rx_ey_locations[5])/2]
    empy_casing_args = {'src':[0,0,zs],
                        'rec':rx_locations_ex_average,
                        'depth':[0],
                        'res':[1e20,1/background_conductivity],
                        'freqtime':frequency,
                        'ab':13,
                        'verb':0,
                        'epermH':[0,1],
                        'epermV':[0,1]}

    # compute field due to wire
    wire_moment = segment_lengths*wire_current
    all_field_x_wire = bipole(**empy_wire_args)
    field_x_wire = np.dot(all_field_x_wire,wire_moment)
    empy_wire_args['rec'] = rx_ey_locations
    all_field_y_wire = bipole(**empy_wire_args)
    field_y_wire = np.dot(all_field_y_wire,wire_moment)

    # compute field due to casing
    field_x_casing = 1j*np.zeros(len(rx_ex_locations[0]))
    field_y_casing = 1j*np.zeros(len(rx_ey_locations[0]))
    for iz in range(len(zs)):
        empy_casing_args['src'] = [0,0,zs[iz]]
        # x
        empy_casing_args['rec'] = rx_locations_ex_average
        empy_casing_args['ab'] = 13
        field_x_casing += casing_moment[iz]*dipole(**empy_casing_args)
        # y
        empy_casing_args['rec'] = rx_locations_ey_average
        empy_casing_args['ab'] = 23
        field_y_casing += casing_moment[iz]*dipole(**empy_casing_args)

    # sum all fields
    field_x = field_x_casing + field_x_wire
    field_y = field_y_casing + field_y_wire

    return(field_x,field_y,field_x_wire,field_y_wire,field_x_casing,field_y_casing)


