import pynbody
import m_ahf
import numpy
import m_halo

def moment_of_inertia(obiect):
    
    mass = obiect['mass']
    
    x = obiect['x']
    y = obiect['y']
    z = obiect['z']
    
    x = x-numpy.sum(x*mass)/numpy.sum(mass)
    y = y-numpy.sum(y*mass)/numpy.sum(mass)
    z = z-numpy.sum(z*mass)/numpy.sum(mass)
    
    Ixx = numpy.sum(mass * (y**2 + z**2))
    Iyy = numpy.sum(mass * (x**2 + z**2))
    Izz = numpy.sum(mass * (x**2 + y**2))
    
    Ixy = numpy.sum(mass*x*y)
    Ixz = numpy.sum(mass*x*z)
    Iyz = numpy.sum(mass*y*z)
    
    return numpy.matrix([[Ixx, -Ixy, -Ixz],[-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])

def com_extract(object_reference):
    r_x = numpy.sum(object_reference['x']*object_reference['mass'])/numpy.sum(object_reference['mass'])
    r_y = numpy.sum(object_reference['y']*object_reference['mass'])/numpy.sum(object_reference['mass'])
    r_z = numpy.sum(object_reference['z']*object_reference['mass'])/numpy.sum(object_reference['mass'])
    
    v_x = numpy.sum(object_reference['vx']*object_reference['mass'])/numpy.sum(object_reference['mass'])
    v_y = numpy.sum(object_reference['vy']*object_reference['mass'])/numpy.sum(object_reference['mass'])
    v_z = numpy.sum(object_reference['vz']*object_reference['mass'])/numpy.sum(object_reference['mass'])
    
    pos = numpy.array([r_x, r_y, r_z])
    vel = numpy.array([v_x, v_y, v_z])
    return pos, vel

def tanslate_coordinates(obiect, pos, vel):
    obiect['pos'] = obiect['pos'] - numpy.array(new_center)
    obiect['vel'] = obiect['vel'] - numpy.array(com_vel)
    
    
def rotate(rotation_matrix, object_t):
    
        Xr = rotation_matrix


        dr = numpy.matrix(object_t['pos'])
        dv = numpy.matrix(object_t['vel'])
        
        #print dx
        #print dx.T

        dr_rot = Xr * dr.T
        dv_rot = Xr * dv.T
        object_t['pos'] = numpy.array(dr_rot.T)
        object_t['vel'] = numpy.array(dv_rot.T)

def i_align(object_reference, object_transformed, align_coef = 0.001):
    pos, vel = com_extract(object_reference)
    translate_coordinates(object_transformed, pos, vel)
    translate_coordinates(object_reference,   pos, vel)
    inu = 0
    ao = 1.
    bo = 1.

    while ao < bo/align_coef:
    
        I_r = moment_of_inertia(object_reference)

        I_r= numpy.matrix(I_r)

        eigvals, eigvecs = numpy.linalg.eig(I_r)
        first = []
        second = []
        third = []

        ordered_eigvals = []
        ordered_eigvecs = []

        eigvals = numpy.array(eigvals)
        eigvecs = numpy.array(eigvecs)
        
        for i in [0,1,2]:
            if eigvals[i] == max(eigvals):
                first = [eigvals[i], eigvecs[i]]
            elif eigvals[i] == min(eigvals):
                third = [eigvals[i], eigvecs[i]]
            else:
                second = [eigvals[i], eigvecs[i]]
    
        ordered_eigvals = [first[0], second[0], third[0]]
    
        ordered_eigvecs = [first[1], second[1], third[1]]

        #print ordered_eigvecs


        Xr = numpy.matrix(ordered_eigvecs)


        rotate(Xr, object_reference)
        rotate(Xr, object_transformed)

        I_rn = moment_of_inertia(object_reference)

        I_rn= numpy.matrix(I_rn)

        values_Irn = numpy.array(I_rn)
        ao = min(numpy.fabs(values_Irn[0][0]),numpy.fabs(values_Irn[1][1]),numpy.fabs(values_Irn[2][2]))
        bo = max(numpy.fabs(values_Irn[0][1]),numpy.fabs(values_Irn[1][2]),numpy.fabs(values_Irn[0][2]))
        inu =inu+1
 
    print "Alignment done in ", inu, "steps"
    print "Alignment coefficient ", bo/ao



