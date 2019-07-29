import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK, pointsToVTK
import pandas as pd

class dipole(object):
    '''
    dipole class
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
        self.symmetry = 1 # 0: no symmetry; 1: periodicity; 2: stellarator symmetry (+periodicity)
        if 'ox' in kwargs:
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
        return

    @classmethod
    def read_dipole(cls, filename, **kwargs):
        '''
        read diploes from FOCUS format (new)
        '''
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
        print('Read {:d} dipoles from {:}'.format(len(ox), filename))
        return cls(ox=ox, oy=oy, oz=oz, Ic=Ic, mm=mm, Lc=Lc, mp=mp, mt=mt, pho=pho, momentq=momentq)

    @classmethod
    def read_dipole_old(cls, filename, zeta=0.0, zeta1=np.pi*2, **kwargs):
        '''
        read diploes from FOCUS format (old)
        '''
        ox = []; oy = []; oz = [];
        mm = []; mp = []; mt = [];
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

    def write_dipole(self, filename, unique=False):
        '''
        write diploes from FOCUS format
        '''
        if not self.sp_switch:
            self.xyz2sp()
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
                    wfile.write("#  Lc  ox   oy   oz  Ic  I  mt  mp \n")
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
    
    def write_vtk(self, vtkname, dim=(1), **kwargs):
        if not self.xyz_switch:
            self.sp2xyz() 
        dim = np.atleast_1d(dim)
        if len(dim) == 1: # save as points
            print("write VTK as points")
            data={"mx":self.mx, "my":self.my, "mz":self.mz}
            if not self.old:
                data.update({"rho":self.pho**self.momentq})
            data.update(kwargs)
            pointsToVTK(vtkname, self.ox, self.oy, self.oz, data=data)#.update(kwargs))
        else : # if manually close the gap
            assert len(dim)==3
            print("write VTK as closed surface")
            phi = 2*np.pi/self.nfp
            def map_toroidal(vec):
                rotate = np.array([[  np.cos(phi), np.sin(phi), 0], \
                                   [ -np.sin(phi), np.cos(phi), 0], \
                                   [                0,               0, 1]])
                return np.matmul(vec, rotate)
            nr, nz, nt = dim
            data_array = {"ox":self.ox, "oy":self.oy, "oz":self.oz, \
                          "mx":self.mx, "my":self.my, "mz":self.mz, \
                          "Ic":self.Ic, "rho":self.pho**self.momentq}
            data_array.update(kwargs)
            for key in list(data_array.keys()):
                new_vec = np.zeros((nr, nz+1, nt+1))
                for ir in range(nr):
                    new_vec[ir,:,:] = map_matrix(np.reshape(data_array[key], dim)[ir,:,:])
                data_array[key] = new_vec
            ox = np.copy(data_array['ox'])
            oy = np.copy(data_array['oy'])
            oz = np.copy(data_array['oz'])   
            del data_array['ox']
            del data_array['oy']
            del data_array['oz'] 
            if self.nfp>1 :
                for ir in range(nr):
                    xyz = map_toroidal(np.transpose([ox[ir,0,:], oy[ir,0,:], oz[ir,0,:]]))
                    ox[ir,nz,:] = xyz[:,0]
                    oy[ir,nz,:] = xyz[:,1]
                    oz[ir,nz,:] = xyz[:,2]
                    moment = map_toroidal(np.transpose([data_array['mx'][ir,0,:],
                                                        data_array['my'][ir,0,:], 
                                                        data_array['mz'][ir,0,:]]))
                    data_array['mx'][ir,nz,:] = moment[:,0]
                    data_array['my'][ir,nz,:] = moment[:,1]
                    data_array['mz'][ir,nz,:] = moment[:,2]
            gridToVTK(vtkname, ox, oy, oz, pointData=data_array)
            return
            ox = np.zeros((nr, nz+1, nt+1))
            oy = np.zeros_like(ox)
            oz = np.zeros_like(ox)
            mx = np.zeros_like(ox)
            my = np.zeros_like(ox)
            mz = np.zeros_like(ox)
            rho = np.zeros_like(ox)
            Ic = np.zeros_like(ox)
            for ir in range(nr):
                ox[ir,:,:] = map_matrix(np.reshape(self.ox, dim)[ir,:,:])
                oy[ir,:,:] = map_matrix(np.reshape(self.oy, dim)[ir,:,:])
                oz[ir,:,:] = map_matrix(np.reshape(self.oz, dim)[ir,:,:])
                mx[ir,:,:] = map_matrix(np.reshape(self.mx, dim)[ir,:,:])
                my[ir,:,:] = map_matrix(np.reshape(self.my, dim)[ir,:,:])
                mz[ir,:,:] = map_matrix(np.reshape(self.mz, dim)[ir,:,:])
                rho[ir,:,:] = map_matrix(np.reshape(self.pho**self.momentq, dim)[ir,:,:])
                Ic[ir,:,:] = map_matrix(np.reshape(self.Ic, dim)[ir,:,:])  
                if self.nfp == 1:
                    continue # map_matrix is enough for 1 period
                # correct toroidal direction
                xyz = map_toroidal(np.transpose([ox[ir,0,:], oy[ir,0,:], oz[ir,0,:]]))
                ox[ir,nz,:] = xyz[:,0].copy()
                oy[ir,nz,:] = xyz[:,1].copy()
                oz[ir,nz,:] = xyz[:,2].copy()
                # correct toroidal direction
                moment = map_toroidal(np.transpose([mx[ir,0,:], my[ir,0,:], mz[ir,0,:]]))
                mx[ir,nz,:] = moment[:,0].copy()
                my[ir,nz,:] = moment[:,1].copy()
                mz[ir,nz,:] = moment[:,2].copy()
            data = {"mx":mx, "my":my, "mz":mz, "rho":rho, "Ic":Ic}
            data.update(kwargs)
            gridToVTK(vtkname, ox, oy, oz, pointData=data)
        return

    def full_period(self, nfp=1):
        """
        map from one period to full periods
        """
        assert nfp>=1
        self.nfp = nfp
        if not self.xyz_switch:
            self.sp2xyz() 
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

    def change_momentq(self, newq):
        """
        change the q factor for density function
        """
        assert newq>0
        pho = self.pho**self.momentq
        self.momentq = newq
        self.pho = np.power(pho, 1.0/newq)
        return

    def plot_pho_profile(self, nrange=10, nofigure=False, **kwargs):
        pho = self.pho**self.momentq
        zone = np.linspace(0, 1, nrange+1, endpoint=True)
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
            plt.bar(zone[:-1], count/float(self.num), width=1.0/nrange, **kwargs)
            ax.set_xlabel('rho', fontsize=15)
            ax.set_ylabel('fraction', fontsize=15)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        return count

    def __del__(self):
        class_name = self.__class__.__name__
