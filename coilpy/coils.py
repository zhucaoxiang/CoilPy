import numpy as np
u0_d_4pi = 1.0E-7
class SingleCoil(object):
    '''
    Single coil class represented as discrete points in Cartesian coordinates
    '''
    def __init__(self, x=[], y=[], z=[], I=0.0, name='coil1', group=1):
        assert len(x) == len(y) == len(z), "dimension not consistent"
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        self.I = I
        self.name = name
        self.group = group
        self.xt = None
        self.yt = None
        self.zt = None
        return
    
    def __del__(self):
        class_name = self.__class__.__name__
        #print(class_name, "destroyed")

    def plot(self, engine='mayavi', **kwargs):
        '''
        plot coil as a line
        '''
        if engine == 'pyplot':
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            # plot in matplotlib.pyplot
            if plt.get_fignums():
                fig = plt.gcf()
                ax = plt.gca()
            else :
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            ax.plot(self.x, self.y, self.z, **kwargs)
        elif engine == 'mayavi':
            # plot 3D line in mayavi.mlab
            from mayavi import mlab # to overrid plt.mlab
            mlab.plot3d(self.x, self.y, self.z, **kwargs)
        else:
            raise ValueError('Invalid engine option {pyplot, mayavi, noplot}')
        return        

    def rectangle(self, width=0.1, height=0.1, frame='centroid', **kwargs):
        '''
        This function expand single coil filament to a rectangle coil;
        
        width
        height
        winding : winding surface data;
        tol: root find tolarence
        '''
        n = np.size(self.x)
        dt = 2*np.pi/(n-1)
        # calculate the tangent 
        if self.xt is None:
            self.spline_tangent()
        xt = self.xt
        yt = self.yt
        zt = self.zt
        # xt = np.gradient(self.x)/dt
        # yt = np.gradient(self.y)/dt
        # zt = np.gradient(self.z)/dt
        tt = np.sqrt(xt*xt + yt*yt + zt*zt)
        xt = xt/tt
        yt = yt/tt
        zt = zt/tt

        # use surface normal if needed
        if frame == 'centroid':
            # use the geometry center is a good idea
            center_x = np.average(self.x[0:n-1])
            center_y = np.average(self.y[0:n-1])
            center_z = np.average(self.z[0:n-1])
            xn = self.x - center_x
            yn = self.y - center_y
            zn = self.z - center_z
            nt = xn*xt + yn*yt + zn*zt
            xn = xn - nt*xt
            yn = yn - nt*yt
            zn = zn - nt*zt
        elif frame == 'frenet':
            self.spline_tangent(der=2)
            xn = self.xa
            yn = self.ya
            zn = self.za
        elif frame == 'parallel':
            # parallel transport frame 
            # Hanson & Ma, Parallel Transp ort Approach to Curve Framing, 1995
            def rotate(x, ang):
                c = np.cos(ang)
                s = np.sin(ang)
                return [[c + x[0]**2*(1-c), x[0]*x[1]*(1-c)-s*x[2], x[2]*x[0]*(1-c)+s*x[1]],
                        [x[0]*x[1]*(1-c)+s*x[2], c+x[1]**2*(1-c), x[2]*x[1]*(1-c)-s*x[0]],
                        [x[0]*x[2]*(1-c)-s*x[1], x[1]*x[2]*(1-c)+s*x[0], c+x[2]**2*(1-c)]]
            T = np.transpose([self.xt, self.yt, self.zt])
            T = T/np.linalg.norm(T, axis=1)[:, np.newaxis]
            B = np.cross(T[:-1], T[1:], axis=1)
            B = B/np.linalg.norm(B, axis=1)[:, np.newaxis]
            theta = np.arccos(np.sum(T[:-1]*T[1:], axis=1))
            V = np.zeros_like(T)
            vx = kwargs['vx']
            vy = kwargs['vy']
            vz = -(vx*T[0,0] + vy*T[0,1])/T[0,2]
            vv = np.linalg.norm([vx, vy, vz])
            V[0,:] = [vx/vv, vy/vv, vz/vv]
            print(np.dot(V[0,:], T[0,:]))
            for i in range(len(theta)):
                V[i+1,:] = rotate(B[i,:], theta[i]) @ V[i,:]
            xn = V[:,0]
            yn = V[:,1]
            zn = V[:,2]
        else:
            assert True, "not finished"
        
        nn = np.sqrt(xn*xn + yn*yn + zn*zn)
        xn = xn/nn
        yn = yn/nn
        zn = zn/nn
        # calculate the bi-normal
        xb = yt*zn - yn*zt
        yb = zt*xn - zn*xt
        zb = xt*yn - xn*yt
        bb = np.sqrt(xb*xb + yb*yb + zb*zb)
        xb = xb/bb
        yb = yb/bb
        zb = zb/bb
        # get the boundary lines
        z1 = self.z - width/2*zb + height/2*zn
        x1 = self.x - width/2*xb + height/2*xn
        x2 = self.x + width/2*xb + height/2*xn
        y2 = self.y + width/2*yb + height/2*yn
        z2 = self.z + width/2*zb + height/2*zn
        x3 = self.x + width/2*xb - height/2*xn
        y3 = self.y + width/2*yb - height/2*yn
        z3 = self.z + width/2*zb - height/2*zn
        x4 = self.x - width/2*xb - height/2*xn
        y4 = self.y - width/2*yb - height/2*yn
        z4 = self.z - width/2*zb - height/2*zn
        y1 = self.y - width/2*yb + height/2*yn
        # assemble 
        xx = np.array([x1, x2, x3, x4, x1])
        yy = np.array([y1, y2, y3, y4, y1])
        zz = np.array([z1, z2, z3, z4, z1])

        return xx, yy, zz
        
    def interpolate(self, num=256):
        '''
        Interpolate to increase more data points
        '''
        from scipy.interpolate import interp1d
        cur_len = len(self.x)
        assert cur_len > 0
        theta = np.linspace(0, 1, num=cur_len, endpoint=True)
        theta_new = np.linspace(0, 1, num=num, endpoint=True)
        for xyz in [self.x, self.y, self.z]:
            f = interp1d(theta, xyz, kind='cubic')
            xyz[:] = f(theta_new)
        return
    
    def magnify(self, ratio):
        """
        magnify coil with a ratio
        """
        # number of points
        nseg = len(self.x)
        # assuming closed curve; should be revised
        if True: #abs(self.x[0] - self.x[-1]) < 1.0E-8:
            nseg -= 1
        assert nseg>1
        # get centroid
        centroid = np.array([np.sum(self.x[0:nseg])/nseg, 
                             np.sum(self.y[0:nseg])/nseg,
                             np.sum(self.z[0:nseg])/nseg])
        # magnify
        for i in range(nseg):
            xyz = np.array([self.x[i], self.y[i], self.z[i]])
            dr = xyz-centroid
            [self.x[i], self.y[i], self.z[i]] = centroid + ratio*dr
        try:
            self.x[nseg] = self.x[0]
            self.y[nseg] = self.y[0]
            self.z[nseg] = self.z[0]
            return
        except:
            return

    def bfield_HH(self, pos, **kwargs):
        """Calculate B field at an arbitrary point using the Hanson-Hirshman expression

        Arguments:
            pos {array-like} -- Cartesian coordinates for the evaluation point

        Returns:
            ndarray -- the calculated magnetic field vector
        """        
        xyz = np.array([self.x, self.y, self.z]).T
        pos = np.atleast_2d(pos)
        assert (pos.shape)[1]  == 3
        Rvec = pos[:,np.newaxis,:] - xyz[np.newaxis,:,:]
        assert (Rvec.shape)[-1] == 3
        RR = np.linalg.norm(Rvec, axis=2)
        Riv = Rvec[:, :-1, :]
        Rfv = Rvec[:, 1:, :]
        Ri = RR[:,:-1]
        Rf = RR[:,1:]
        B = np.sum(np.cross(Riv, Rfv)*((Ri+Rf)/((Ri*Rf)*(Ri*Rf+np.sum(Riv*Rfv, axis=2))))[:, :, np.newaxis], axis=1)\
            *u0_d_4pi*self.I
        return B

    def bfield(self, pos):        
        ob_pos = np.atleast_1d(pos)
        dx = ob_pos[0] - self.x[:-1]
        dy = ob_pos[1] - self.y[:-1]
        dz = ob_pos[2] - self.z[:-1]
        dr = dx*dx + dy*dy +dz*dz
        Bx = (dz*self.yt[:-1] - dy*self.zt[:-1])*np.power(dr, -1.5)
        By = (dx*self.zt[:-1] - dz*self.xt[:-1])*np.power(dr, -1.5)
        Bz = (dy*self.xt[:-1] - dx*self.yt[:-1])*np.power(dr, -1.5)
        B = np.array([np.sum(Bx), np.sum(By), np.sum(Bz)])*self.dt*u0_d_4pi*self.I
        return B        
    
    def spline_tangent(self, order=3, der=1):
        """Calculate the tangent of coil using spline interpolation

        Args:
            order (int, optional): Order of spline interpolation used. Defaults to 3.
        """        
        from scipy import interpolate
        t = np.linspace(0, 2*np.pi, len(self.x), endpoint=True)
        self.dt = 2*np.pi/(len(self.x)-1)
        fx = interpolate.splrep(t, self.x, s=0, k=order)
        fy = interpolate.splrep(t, self.y, s=0, k=order)
        fz = interpolate.splrep(t, self.z, s=0, k=order)
        self.xt = interpolate.splev(t, fx, der=1)
        self.yt = interpolate.splev(t, fy, der=1)
        self.zt = interpolate.splev(t, fz, der=1)
        if der == 2:
            self.xa = interpolate.splev(t, fx, der=2)
            self.ya = interpolate.splev(t, fy, der=2)
            self.za = interpolate.splev(t, fz, der=2)
        return 

    def fourier_tangent(self):
        from .misc import fft_deriv
        self.dt = 2*np.pi/(len(self.x)-1)
        fftxy = fft_deriv(self.x[:-1]+1j*self.y[:-1])
        fftz = fft_deriv(self.z[:-1])
        self.xt = np.real(fftxy)
        self.yt = np.imag(fftxy)
        self.zt = np.real(fftz)
        self.xt = np.concatenate((self.xt, self.xt[0:1]))
        self.yt = np.concatenate((self.yt, self.yt[0:1]))
        self.zt = np.concatenate((self.zt, self.zt[0:1]))
        return         

    def toVTK(self, vtkname, **kwargs):
        """Write a VTK file

        Args:
            vtkname (string): VTK filename
        """        
        from pyevtk.hl import polyLinesToVTK
        kwargs.setdefault('cellData', {})
        kwargs['cellData'].setdefault('I', np.array(elf.I))
        polyLinesToVTK(vtkname, np.array(self.x), np.array(self.y), np.array(self.z),
                       np.array([len(self.x)]), **kwargs)
        return
        
class Coil(object):
    def __init__(self, xx=[], yy=[], zz=[], II=[], names=[], groups=[]):
        assert len(xx) == len(yy) == len(zz) == len(II) == len(names) == len(groups), "dimension not consistent"
        self.num = len(xx)
        self.data = []
        for i in range(self.num):
            self.data.append(SingleCoil(x=xx[i], y=yy[i], z=zz[i], I=II[i], name=names[i], group=groups[i]))
        self.index = 0
        return 

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.num:
            self.index += 1
            return self.data[self.index-1]
        else:
            self.index = 0 
            raise StopIteration()

    def __len__(self):
        return self.num

    @classmethod
    def read_makegrid(cls, filename):
        """read MAKEGRID format 
        Args:
            filename (str) : file path and name
        Returns:
            Coil class
        """
        import os
        # check existence
        if not os.path.exists(filename) :
            raise IOError ("File not existed. Please check again!")
        # read and parse data
        cls.header = ''
        with open(filename,'r') as coilfile: #read coil xyz and I
            cls.header = ''.join((coilfile.readline(), coilfile.readline(), coilfile.readline()))
            icoil = 0
            xx = [[]]; yy = [[]]; zz = [[]]
            II = []; names = []; groups = []
            tmpI = 0.0
            for line in coilfile:
                linelist = line.split()
                if len(linelist) < 4 :
                    #print("End of file or invalid format!")                    
                    break
                xx[icoil].append(float(linelist[0]))
                yy[icoil].append(float(linelist[1]))
                zz[icoil].append(float(linelist[2]))
                if len(linelist) == 4 :
                    tmpI = float(linelist[-1])
                if len(linelist) > 4 :
                    II.append(tmpI)
                    try:
                        group = int(linelist[4])
                    except:
                        group = len(groups) + 1
                    groups.append(group)
                    try:
                        name = linelist[5]
                    except IndexError:
                        name = 'coil'
                    names.append(name)
                    icoil = icoil + 1
                    xx.append([]); yy.append([]); zz.append([])
        xx.pop(); yy.pop(); zz.pop()
        # print(len(xx) , len(yy) , len(zz) , len(II) , len(names) , len(groups))
        return cls(xx=xx, yy=yy, zz=zz, II=II, names=names, groups=groups)

    def plot(self, irange=[], **kwargs):
        """Plot coils in mayavi or matplotlib  

        Args:
            irange (list, optional): Coil list to be plotted. Defaults to [].
        """        
        if len(irange)==0:
            irange = range(self.num)    
        for i in irange:
            self.data[i].plot(**kwargs)
        return

    def save_makegrid(self, filename, nfp=1, **kwargs):
        """write in MAKEGRID format

        Arguments:
            filename {str} -- file name and path
            nfp {int} -- number of toroidal periodicity (default: 1)
        """
        assert len(self) > 0
        with open(filename, 'w') as wfile :
            wfile.write("periods {:3d} \n".format(nfp))
            wfile.write("begin filament \n")
            wfile.write("mirror NIL \n")
            for icoil in list(self):
                Nseg = len(icoil.x) # number of segments;
                assert Nseg > 1
                for iseg in range(Nseg-1): # the last point match the first one;
                    wfile.write("{:15.7E} {:15.7E} {:15.7E} {:15.7E}\n".format(
                        icoil.x[iseg], icoil.y[iseg], icoil.z[iseg], icoil.I))
                wfile.write("{:15.7E} {:15.7E} {:15.7E} {:15.7E} {:} {:10} \n".format(
                    icoil.x[0], icoil.y[0], icoil.z[0], 0.0, icoil.group, icoil.name))
            wfile.write("end \n")
        return

    def toVTK(self, vtkname, **kwargs):
        """Write entire coil set into a VTK file

        Args:
            vtkname (str): VTK filename
            kwargs (dict): Optional kwargs passed to "polyLinesToVTK"
        """        
        from pyevtk.hl import polyLinesToVTK
        currents = []
        groups = []
        x = []
        y = []
        z = []
        lx = []
        for icoil in list(self):
            currents.append(icoil.I)
            groups.append(icoil.group)
            x.append(icoil.x)
            y.append(icoil.y)
            z.append(icoil.z)
            lx.append(len(icoil.x))
        kwargs.setdefault('cellData', {})
        kwargs['cellData'].setdefault('I', np.array(currents))
        kwargs['cellData'].setdefault('Igroup', np.array(groups))
        polyLinesToVTK(vtkname, np.concatenate(x), np.concatenate(y), np.concatenate(z),
                        np.array(lx), **kwargs)
        return

    def __del__(self):
        class_name = self.__class__.__name__
