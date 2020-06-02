import numpy as np
from .misc import xy2rp, map_matrix, toroidal_period

class Dipole(object):
    '''
    magnetic dipole class
    '''
    def __init__(self, **kwargs):
        '''
        Initialize empty class
        '''
        self.num = 0 # total number of dipoles
        self.nfp = 1 # toroidal_periodicity
        self.momentq = 1 # q expotent
        self.sp_switch = False # switch to indicate if using spherical coordinates
        self.xyz_switch = False # switch to indicate if using spherical coordinates
        self.old = False # old format or new
        self.symmetry = 2 # 0: no symmetry; 1: periodicity; 2: stellarator symmetry (+periodicity)
        if 'mm' in kwargs: # spherical 
            self.ox = kwargs['ox']
            self.oy = kwargs['oy']
            self.oz = kwargs['oz']
            self.mm = kwargs['mm']
            self.mt = kwargs['mt']
            self.mp = kwargs['mp']
            self.pho = kwargs['pho']
            self.momentq = kwargs['momentq']
            self.Ic = kwargs['Ic']
            self.Lc = kwargs['Lc']
            self.num = len(self.ox)
            self.sp_switch = True
            self.name = kwargs.get('name', 'dipole')
            self.rho = self.pho**self.momentq
        elif 'mx' in kwargs:
            self.ox = kwargs['ox']
            self.oy = kwargs['oy']
            self.oz = kwargs['oz']
            self.mx = kwargs['mx']
            self.my = kwargs['my']
            self.mz = kwargs['mz']
            self.mm = np.sqrt(self.mx**2 + self.my**2 + self.mz**2)
            self.num = len(self.ox)
            self.pho = np.ones(self.num)
            self.Ic = np.zeros(self.num, dtype=int)
            self.Lc = np.zeros(self.num, dtype=int)        
            self.xyz_switch = True
            self.name = kwargs.get('name', 'dipole')
            self.rho = np.ones(self.num)
        return

    @classmethod
    def open(cls, filename, verbose=False, **kwargs):
        '''
        read diploes from FOCUS format (new)
        '''
        import pandas as pd
        with open(filename, 'r') as coilfile:
            coilfile.readline()
            line = coilfile.readline().split()
            try:
                momentq = int(line[1])
            except:
                print("Moment Q factor was not read. Default=1.")
                momentq = 1
        data = pd.read_csv(filename, skiprows=3, header=None)
        ox = np.array(data[3], dtype=float)
        oy = np.array(data[4], dtype=float)
        oz = np.array(data[5], dtype=float)
        Ic = np.array(data[6], dtype=int)
        mm = np.array(data[7], dtype=float)
        pho= np.array(data[8], dtype=float)
        Lc = np.array(data[9], dtype=int)
        mp = np.array(data[10], dtype=float)
        mt = np.array(data[11], dtype=float)
        if verbose:
            print('Read {:d} dipoles from {:}'.format(len(ox), filename))
        return cls(ox=ox, oy=oy, oz=oz, Ic=Ic, mm=mm, Lc=Lc, mp=mp, mt=mt, pho=pho, momentq=momentq, name=filename)

    @classmethod
    def read_dipole_old(cls, filename, zeta=0.0, zeta1=np.pi*2, **kwargs):
        '''
        read diploes from FOCUS format (old)
        '''
        ox = []; oy = []; oz = []
        mm = []; mp = []; mt = []
        Ic = []; Lc = []
        with open(filename, 'r') as coilfile:
            coilfile.readline()
            Ncoils = int(coilfile.readline())
            for icoil in range(Ncoils):
                coilfile.readline()
                coilfile.readline()
                linelist = coilfile.readline().split()
                if int(linelist[0]) == 3 : # central current and background Bz
                    coilfile.readline()
                    linelist = coilfile.readline().split() 
                elif int(linelist[0]) == 2 : # dipoles
                    coilfile.readline()
                    linelist = coilfile.readline().split()
                    r, phi = xy2rp(float(linelist[1]), float(linelist[2]))
                    if phi>=zeta and phi<=zeta1:
                        Lc.append(  int(linelist[0]))
                        ox.append(float(linelist[1]))
                        oy.append(float(linelist[2]))
                        oz.append(float(linelist[3]))
                        Ic.append(  int(linelist[4]))
                        mm.append(float(linelist[5]))
                        mt.append(float(linelist[6]))
                        mp.append(float(linelist[7]))
                elif int(linelist[0]) == 1 : # Fourier coils
                    for i in range(11):
                        coilfile.readline()
                else :
                    raise ValueError('Invalid coiltype = {:d}.'.format(int(linelist[0])))
        nc = len(ox)
        if nc == 0 :
            print('Warning: no dipoles was read from '+filename)
            return
        print('Read {:d} dipoles from {:}. Please manually set self.old=True.'.format(nc, filename))
        ox = np.array(ox)
        oy = np.array(oy)
        oz = np.array(oz)
        mm = np.array(mm)
        mt = np.array(mt)
        mp = np.array(mp)
        Lc = np.array(Lc)
        Ic = np.array(Ic)
        pho = np.ones_like(ox)
        momentq = 1
        return cls(ox=ox, oy=oy, oz=oz, Ic=Ic, mm=mm, Lc=Lc, mp=mp, mt=mt, pho=pho, momentq=momentq)

    def sp2xyz(self):
        '''
        spherical coordinates to cartesian coordinates
        '''
        assert self.sp_switch == True, "You are not using spherical coordinates"
        if self.old:
            self.mx = self.mm * np.sin(self.mt) * np.cos(self.mp) 
            self.my = self.mm * np.sin(self.mt) * np.sin(self.mp) 
            self.mz = self.mm * np.cos(self.mt)
        else :
            self.mx = self.mm * self.pho**self.momentq * np.sin(self.mt) * np.cos(self.mp) 
            self.my = self.mm * self.pho**self.momentq * np.sin(self.mt) * np.sin(self.mp) 
            self.mz = self.mm * self.pho**self.momentq * np.cos(self.mt)
        self.xyz_switch = True
        return

    def xyz2sp(self):
        '''
        cartesian coordinates to spherical coordinates
        '''
        assert self.xyz_switch == True, "You are not using cartesian coordinates"
        if self.old :
            self.mm = np.sqrt(self.mx*self.mx + self.my*self.my + self.mz*self.mz)
            #self.pho = np.ones_like(self.mm)
            #self.momentq = 1
        else :
            self.pho = np.power(np.sqrt(self.mx*self.mx + self.my*self.my + self.mz*self.mz)/self.mm, 1.0/self.momentq)
        self.mp = np.arctan2(self.my, self.mx)
        self.mt = np.arccos(self.mz/(self.mm*self.pho**self.momentq))
        self.sp_switch = True
        return   

    def save(self, filename, unique=False):
        '''
        write diploes from FOCUS format
        '''
        if not self.sp_switch:
            self.xyz2sp()
        print('symmetry : {:d}'.format(self.symmetry))
        with open(filename, 'w') as wfile :
            wfile.write(" # Total number of coils,  momentq \n")
            if unique:
                wfile.write("{:6d},  {:4d}\n".format(self.num/self.nfp, self.momentq))
            else :
                wfile.write("{:6d},  {:4d}\n".format(self.num, self.momentq))
            if self.old:
                for icoil in range(self.num):
                    if unique:
                        if np.mod(icoil, self.nfp)==0:
                            continue
                    wfile.write("#-----------------{}---------------------------\n".format(icoil+1))
                    wfile.write("#coil_type     coil_name \n")
                    wfile.write("   {:3d}  {:1d}  pm_{:010d}\n".format(2, 1, icoil+1))
                    wfile.write("#  Lc  ox   oy   oz  Ic  I  mp  mt \n")
                    wfile.write("{:6d} {:23.15E} {:23.15E} {:23.15E} {:6d} {:23.15E} {:23.15E} {:23.15E}\n"\
                       .format(self.Lc[icoil], self.ox[icoil], self.oy[icoil], self.oz[icoil], \
                                   self.Ic[icoil], self.mm[icoil], self.mt[icoil], self.mp[icoil] ))
            else:
                wfile.write('#coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt \n')
                for i in range(self.num):
                    if unique:
                        if np.mod(i, self.nfp)==0:
                            continue
                    wfile.write(" 2, {:1d}, pm_{:010d}, {:15.8E}, {:15.8E}, {:15.8E}, {:2d}, {:15.8E}," \
                   "{:15.8E}, {:2d}, {:15.8E}, {:15.8E} \n".format(self.symmetry, i, self.ox[i], self.oy[i], self.oz[i], 
                   self.Ic[i], self.mm[i], self.pho[i], self.Lc[i], self.mp[i], self.mt[i]))
        return
    
    def toVTK(self, vtkname=None, dim=(1), close=True, ntnz=False, toroidal=False, **kwargs):
        from pyevtk.hl import gridToVTK, pointsToVTK
        if not self.xyz_switch:
            self.sp2xyz()
        if vtkname is None:
            vtkname = self.name
        dim = np.atleast_1d(dim)
        if len(dim) == 1: # save as points
            print("write VTK as points")
            data={"m": (self.mx, self.my, self.mz)}
            if not self.old:
                data.update({"rho":self.pho**self.momentq})
            data.update(kwargs)
            pointsToVTK(vtkname, self.ox, self.oy, self.oz, data=data)#.update(kwargs))
        else : # save as surfaces
            assert len(dim)==3
            print("write VTK as closed surface")
            if close:
                # manually close the gap
                phi = 2*np.pi/self.nfp
                def map_toroidal(vec):
                    rotate = np.array([[  np.cos(phi), np.sin(phi), 0], \
                                       [ -np.sin(phi), np.cos(phi), 0], \
                                       [                0,               0, 1]])
                    return np.matmul(vec, rotate)                
                data_array = {"ox":self.ox, "oy":self.oy, "oz":self.oz, \
                              "mx":self.mx, "my":self.my, "mz": self.mz, \
                              "Ic":self.Ic, "rho":self.pho**self.momentq}
                data_array.update(kwargs)        
                nr, nz, nt = dim
                for key in list(data_array.keys()):
                    new_vec = np.zeros((nr, nz+1, nt+1))
                    for ir in range(nr):
                        new_vec[ir,:,:] = map_matrix(np.reshape(data_array[key], dim)[ir,:,:])
                    if toroidal :
                        data_array[key] = new_vec
                    else :
                        if ntnz :
                            data_array[key] = np.ascontiguousarray(new_vec[:,:,:-1])
                        else :
                            data_array[key] = np.ascontiguousarray(new_vec[:,:-1,:])
                ox = np.copy(data_array['ox'])
                oy = np.copy(data_array['oy'])
                oz = np.copy(data_array['oz'])   
                del data_array['ox']
                del data_array['oy']
                del data_array['oz'] 
                data_array['m'] = (data_array['mx'], data_array['mx'], data_array['mx'])
                if toroidal and self.nfp>1 :
                    for ir in range(nr):
                        if ntnz:
                            xyz = map_toroidal(np.transpose([ox[ir,:,0], oy[ir,:,0], oz[ir,:,0]]))
                            ox[ir,:,nz] = xyz[:,0]
                            oy[ir,:,nz] = xyz[:,1]
                            oz[ir,:,nz] = xyz[:,2]
                            moment = map_toroidal(np.transpose([data_array['mx'][ir,:,0],
                                                                data_array['my'][ir,:,0], 
                                                                data_array['mz'][ir,:,0]]))
                            data_array['m'][0][ir,:,nz] = moment[:,0]
                            data_array['m'][1][ir,:,nz] = moment[:,1]
                            data_array['m'][2][ir,:,nz] = moment[:,2]
                        else : 
                            xyz = map_toroidal(np.transpose([ox[ir,0,:], oy[ir,0,:], oz[ir,0,:]]))
                            ox[ir,nz,:] = xyz[:,0]
                            oy[ir,nz,:] = xyz[:,1]
                            oz[ir,nz,:] = xyz[:,2]
                            moment = map_toroidal(np.transpose([data_array['mx'][ir,0,:],
                                                                data_array['my'][ir,0,:], 
                                                                data_array['mz'][ir,0,:]]))
                            data_array['m'][0][ir,nz,:] = moment[:,0]
                            data_array['m'][1][ir,nz,:] = moment[:,1]
                            data_array['m'][2][ir,nz,:] = moment[:,2]
                del data_array['mx']
                del data_array['my']
                del data_array['mz']
                gridToVTK(vtkname, ox, oy, oz, pointData=data_array)
                return
            else:
                ox = np.reshape(self.ox[:self.num], dim)
                oy = np.reshape(self.oy[:self.num], dim)
                oz = np.reshape(self.oz[:self.num], dim)
                mx = np.reshape(self.mx[:self.num], dim)
                my = np.reshape(self.my[:self.num], dim)
                mz = np.reshape(self.mz[:self.num], dim)
                rho = np.reshape(self.pho[:self.num]**self.momentq, dim)
                Ic = np.reshape(self.Ic[:self.num], dim)
            data = {"m": (mx, my, mz), "rho":rho, "Ic":Ic}
            data.update(kwargs)
            gridToVTK(vtkname, ox, oy, oz, pointData=data)
        return

    def full_period(self, nfp=1, symmetry=False, dim=None):
        """
        map from one period to full periods
        """
        assert nfp>=1
        self.nfp = nfp
        if not self.xyz_switch:
            self.sp2xyz()
        # change order
        if dim is not None :
            self.ox = np.ravel(np.transpose(np.reshape(self.ox, dim)[::-1,::-1,::-1], (2,0,1)))
            self.oy = np.ravel(np.transpose(np.reshape(self.oy, dim)[::-1,::-1,::-1], (2,0,1)))
            self.oz = np.ravel(np.transpose(np.reshape(self.oz, dim)[::-1,::-1,::-1], (2,0,1)))
            self.mx = np.ravel(np.transpose(np.reshape(self.mx, dim)[::-1,::-1,::-1], (2,0,1)))
            self.my = np.ravel(np.transpose(np.reshape(self.my, dim)[::-1,::-1,::-1], (2,0,1)))
            self.mz = np.ravel(np.transpose(np.reshape(self.mz, dim)[::-1,::-1,::-1], (2,0,1)))
            self.mm = np.ravel(np.transpose(np.reshape(self.mm, dim)[::-1,::-1,::-1], (2,0,1)))
            self.Ic = np.ravel(np.transpose(np.reshape(self.Ic, dim)[::-1,::-1,::-1], (2,0,1)))
            self.Lc = np.ravel(np.transpose(np.reshape(self.Lc, dim)[::-1,::-1,::-1], (2,0,1)))
            self.pho= np.ravel(np.transpose(np.reshape(self.pho,dim)[::-1,::-1,::-1], (2,0,1)))
        if symmetry:
            # get the stellarator symmetry part first
            # Here, we assume no dipoles on the symmetry plane, or only half are listed.
            self.num *= 2
            self.ox = np.concatenate((self.ox , self.ox[::-1]))
            self.oy = np.concatenate((self.oy , self.oy[::-1]*(-1)))
            self.oz = np.concatenate((self.oz , self.oz[::-1]*(-1)))
            self.mx = np.concatenate((self.mx , self.mx[::-1]*(-1)))
            self.my = np.concatenate((self.my , self.my[::-1]))
            self.mz = np.concatenate((self.mz , self.mz[::-1]))
            self.mm = np.concatenate((self.mm , self.mm[::-1]))
            self.pho= np.concatenate((self.pho, self.pho[::-1]))
            self.Ic = np.concatenate((self.Ic , self.Ic[::-1]))
            self.Lc = np.concatenate((self.Lc , self.Lc[::-1]))
        xyz = toroidal_period(np.transpose([self.ox, self.oy, self.oz]), self.nfp)
        self.ox = xyz[:,0].copy()
        self.oy = xyz[:,1].copy()
        self.oz = xyz[:,2].copy()
        moment = toroidal_period(np.transpose([self.mx, self.my, self.mz]), self.nfp)
        self.mx = moment[:,0].copy()
        self.my = moment[:,1].copy()
        self.mz = moment[:,2].copy()
        self.mm = np.tile(self.mm, self.nfp)
        self.pho = np.tile(self.pho, self.nfp)
        self.Ic = np.tile(self.Ic, self.nfp)
        self.Lc = np.tile(self.Lc, self.nfp)
        self.num *= self.nfp        
        return
    
    def inverse(self):
        if not self.xyz_switch:
            self.sp2xyz()
        self.ox = np.copy(self.ox[::-1])
        self.oy = np.copy(self.oy[::-1]*(-1))
        self.oz = np.copy(self.oz[::-1]*(-1))
        self.mx = np.copy(self.mx[::-1]*(-1))
        self.my = np.copy(self.my[::-1])
        self.mz = np.copy(self.mz[::-1])
        self.mm = np.copy(self.mm[::-1])
        self.pho= np.copy(self.pho[::-1])
        self.Ic = np.copy(self.Ic[::-1])
        self.Lc = np.copy(self.Lc[::-1])
        return
        
    def change_momentq(self, newq):
        """
        change the q factor for density function
        """
        assert newq>0
        pho = self.pho**self.momentq
        # get signs
        self.sp2xyz()
        sign = np.sign(pho)
        if newq%2 == 0: # even number, flip the orientation when negative
            #self.mx *= sign
            #self.my *= sign
            #self.mz *= sign
            self.momentq = newq
            self.xyz2sp()
            # convert to positive rho
            #self.pho = np.power(np.abs(pho), 1.0/newq)
        else: # odd exponetial index
            self.momentq = newq
            # convert to positive rho
            self.pho = np.power(np.abs(pho), 1.0/newq)
            self.pho *= sign
        return

    def plot_rho_profile(self, lower=0, upper=1, nrange=10, nofigure=False, **kwargs):
        import matplotlib.pyplot as plt
        pho = np.abs(self.pho**self.momentq)
        zone = np.linspace(lower, upper, nrange+1, endpoint=True)
        count = []
        for i in range(nrange-1):
            count.append(((pho>=zone[i]) & (pho<zone[i+1])).sum())
        count.append(((pho>=zone[nrange-1]) & (pho<=zone[nrange])).sum())
        count = np.array(count)
        if not nofigure:
            if plt.get_fignums():
                fig = plt.gcf()
                ax = plt.gca()
            else :
                fig, ax = plt.subplots()
            plt.bar(zone[:-1], 100*count/float(self.num), width=0.9/nrange, **kwargs)
            ax.set_xlabel(r'$|\rho|$', fontsize=15)
            ax.set_ylabel('fraction [%]', fontsize=15)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        return count

    def volume(self, magnitization=1.1E6, **kwargs):
        self.total_moment = np.sum(np.abs(self.rho*self.mm))
        return self.total_moment/magnitization

    def orientation(self, unit=True, uniform=False):
        oldq = self.momentq
        self.change_momentq(2) # to a even number
        self.change_momentq(oldq) # recover
        if unit: 
            self.mm[:] = 1.
        if uniform:
            self.pho[:] = 1.
        self.sp2xyz()
        return

    def plot(self, start=0, end=None, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        if end is None:
            end = self.num
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.ox[start:end], self.oy[start:end], self.oz[start:end], **kwargs)
        return ax
        
    def __add__(self, other):
        """ Combine two dipole files together
        
        """
        assert self.momentq == other.momentq
        return Dipole(ox=np.concatenate((self.ox, other.ox)),
                      oy=np.concatenate((self.oy, other.oy)),
                      oz=np.concatenate((self.oz, other.oz)),
                      Ic=np.concatenate((self.Ic, other.Ic)),
                      mm=np.concatenate((self.mm, other.mm)),
                      Lc=np.concatenate((self.Lc, other.Lc)),
                      mp=np.concatenate((self.mp, other.mp)),
                      mt=np.concatenate((self.mt, other.mt)),
                      pho=np.concatenate((self.pho, other.pho)),
                      momentq=self.momentq, 
                      name=self.name + '+' + other.name)
        
    def __del__(self):
        class_name = self.__class__.__name__
