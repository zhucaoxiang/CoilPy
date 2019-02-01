from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab # to overrid plt.mlab

class fourier_surface(object):
    '''
    surface in Fourier representation
    R = \sum RBC cos(mu-nv) + RBS sin(mu-nv)
    Z = \sum ZBC cos(mu-nv) + ZBS sin(mu-nv)
    '''
    def __init__(self, mnmodes=999):
        # initialization
        self.mn = mnmodes # total number of harmonics
        assert self.mn > 0
        self.xm  = np.zeros(self.mn)
        self.xn  = np.zeros(self.mn)
        self.rbc = np.zeros(self.mn)
        self.rbs = np.zeros(self.mn)
        self.zbc = np.zeros(self.mn)
        self.zbs = np.zeros(self.mn)

    def disc_rz(self, zeta=0.0, npoints=360): 
        # discretization in (R,Z) at phi=zeta
        assert npoints > 0
        self.rr = np.zeros(npoints)
        self.zz = np.zeros(npoints)
        theta = np.linspace(0, 2*np.pi, npoints)
        for ipoint in range(npoints):
            tmpr = self.rbc * np.cos(self.xm*theta[ipoint]-self.xn*zeta) \
                 + self.rbs * np.sin(self.xm*theta[ipoint]-self.xn*zeta) 
            self.rr[ipoint] = np.sum(tmpr) #r value at ipont

            tmpz = self.zbc * np.cos(self.xm*theta[ipoint]-self.xn*zeta) \
                 + self.zbs * np.sin(self.xm*theta[ipoint]-self.xn*zeta) 
            self.zz[ipoint] = np.sum(tmpz) #z value at ipont

    def disc_xyz(self, zeta=0.0, zeta1=np.pi*2, npol=360, ntor=360):
        # discretization in (X,Y,Z) between phi=zeta and phi=zeta1
        assert npol > 0 and ntor > 0
        self.xsurf  = np.zeros([npol, ntor]) # xsurf surface elements
        self.ysurf  = np.zeros([npol, ntor])
        self.zsurf  = np.zeros([npol, ntor])
        for i in range(ntor):
            ator = zeta + i*(zeta1-zeta)/(ntor-1) #zeta
            self.disc_rz(zeta=ator, npoints=npol)
            self.xsurf[:,i] = self.rr * np.cos(ator)
            self.ysurf[:,i] = self.rr * np.sin(ator)
            self.zsurf[:,i] = self.zz

    def plot(self, zeta=0.0, npoints=360, color=(1,0,0), style='-', width=2.0,
             label='toroidal surface', marker=None):
        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        else :
            fig, ax = plt.subplots()
        self.disc_rz(zeta=zeta, npoints=npoints)
        ax.plot(self.rr, self.zz, color=color, linewidth=width, linestyle=style, label=label, marker=marker)
        plt.axis('equal')
        plt.xlabel('R [m]',fontsize=20)
        plt.ylabel('Z [m]',fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    
    def plot3d(self, scalar=None, zeta=0.0, zeta1=2*np.pi, color=(1,0,0), npol=360, ntor=360):
        # plot 3D surface in mayavi.mlab
        self.disc_xyz(zeta=zeta, zeta1=zeta1, npol=npol, ntor=ntor)
        mlab.mesh(self.xsurf, self.ysurf, self.zsurf, scalars=scalar, color=color, representation = 'surface')

    def __del__(self):
        class_name = self.__class__.__name__
        
############################

def read_plasma_boundary(filename):
    """
    Read plasma.boundary file in FOCUS format.
    Return a toroidal surface class.
    """
    with open(filename, 'r') as f:
        line = f.readline() #skip one line
        line = f.readline()
        num = int(line.split()[0]) # harmonics number
        nfp = int(line.split()[1]) # number of field periodicity
        nbn = int(line.split()[2]) # number of Bn harmonics
        plas_data = np.zeros(num, dtype=[('n', np.float64),('m',np.float64), # m,n saving as double
                                         ('Rbc',np.float64), ('Rbs', np.float64),
                                         ('Zbc',np.float64), ('Zbs', np.float64)])
        line = f.readline() #skip one line
        line = f.readline() #skip one line
        for i in range(num):
            line = f.readline()
            plas_data[i] = tuple([float(j) for j in line.split()])
        plas_data['n'] *= nfp

        if nbn>0 :
            bn_data = np.zeros(nbn, dtype=[('n', np.float64),('m',np.float64),
                                           ('bnc', np.float64),('bns',np.float64)])
            line = f.readline() #skip one line
            line = f.readline() #skip one line
            for i in range(nbn):
                line = f.readline()
                bn_data[i] = tuple([float(j) for j in line.split()])
            bn_data['n'] *= nfp
        else :
            bn_data = []        
    print("read {} Fourier harmonics for plasma boundary and {} for Bn distribution in {}."\
        .format(num,nbn,filename))

    focus_plasma = fourier_surface(num)
    focus_plasma.nfp = nfp
    focus_plasma.xn[:] = np.array(plas_data['n'][:], np.int)
    focus_plasma.xm[:] = np.array(plas_data['m'][:], np.int)
    focus_plasma.rbc[:] = plas_data['Rbc'][:]
    focus_plasma.rbs[:] = plas_data['Rbs'][:]
    focus_plasma.zbc[:] = plas_data['Zbc'][:]
    focus_plasma.zbs[:] = plas_data['Zbs'][:]

    return focus_plasma #plas_data, bn_data    

############################

def read_vmec_surfce(filename, ns=-1):
    """
    Read VMEC output netcdf file.
    Return a toroidal surface class.
    """
    vmec = ncdata.open_dataset(wout)    
    vmec_plasma = fourier_surface(len(vmec['xm'].values))
    vmec_plasma.nfp = vmec['nfp'].values
    vmec_plasma.xn = vmec['xn'].values
    vmec_plasma.xm = vmec['xm'].values
    rmnc = vmec['rmnc'].values
    zmns = vmec['zmns'].values
    vmec_plasma.rbc = rmnc[ns,:]
    vmec_plasma.zbs = zmns[ns,:]
    if vmec['lasym__logical__'].values: # if no stellarator symmetry
        zmnc = vmec['zmnc'].values
        rmns = vmec['rmns'].values
        vmec_plasma.rbs = rmns[ns,:]
        vmec_plasma.zbc = zmnc[ns,:]
    else :
        vmec_plasma.rbs = np.zeros_like(rbc)
        vmec_plasma.zbc = np.zeros_like(zbs)
    
    return vmec_plasma

############################

def read_vmec_surfce(filename):
    """
    Read NESCOIL output file.
    Return a toroidal surface class.
    """
    with open(filename, 'r') as f:
        line = ''
        while "phip_edge" not in line:
            line = f.readline()
        line = f.readline()
        nfp = int(line.split()[0])
        #print "nfp:",nfp

        line = ''
        while "Current Surface" not in line:
            line = f.readline()
        line = f.readline()
        line = f.readline()
        #print "Number of Fourier modes in coil surface from nescin file: ",line
        num = int(line)
        plas = np.zeros(num, dtype=[('m',np.float64), ('n', np.float64), #m,n saving as double
                                         ('Rbc', np.float64), ('Zbs', np.float64),
                                         ('Rbs', np.float64), ('Zbc', np.float64) ])
        line = f.readline() #skip one line
        line = f.readline() #skip one line
        for i in range(num):
            line = f.readline()
            plas[i] = tuple([float(j) for j in line.split()])
        plas['n'] *= -1   # minus sign since NESCOIL use mu+nv

    wind_surf = fourier_surface(num)
    wind_surf.nfp = nfp
    wind_surf.xn[:] = np.array(plas['n'][:], np.int)
    wind_surf.xm[:] = np.array(plas['m'][:], np.int)
    wind_surf.rbc[:] = plas['Rbc'][:]
    wind_surf.rbs[:] = plas['Rbs'][:]
    wind_surf.zbc[:] = plas['Zbc'][:]
    wind_surf.zbs[:] = plas['Zbs'][:]

    return wind_surf
