import numpy as np 
import matplotlib.pyplot as plt
from .sortedDict import SortedDict

__all__ = ['VMECout']

def cfunct(theta,zeta,fmnc,xm,xn):
    (ns,mn)=fmnc.shape
    lt = len(theta)
    lz = len(zeta)
    mt=np.matmul(np.transpose(np.atleast_2d(xm)),np.atleast_2d(theta))
    nz=np.matmul(np.transpose(np.atleast_2d(xn)),np.atleast_2d(zeta))
    cosmt=np.cos(mt)
    sinmt=np.sin(mt)
    cosnz=np.cos(nz)
    sinnz=np.sin(nz)
    f = np.zeros((ns,lt,lz))
    fmn = np.ndarray((mn,lt))
    for k in range(ns):
        fmn = np.broadcast_to(fmnc[k,:],(lt,mn)).transpose()
        fmncosmt=(fmn*cosmt).transpose()
        fmnsinmt=(fmn*sinmt).transpose()
        f[k,:,:]=np.matmul(fmncosmt, cosnz)-np.matmul(fmnsinmt, sinnz)
    return f
    
def sfunct(theta,zeta,fmnc,xm,xn):
    (ns,mn)=fmnc.shape
    lt = len(theta)
    lz = len(zeta)
    mt=np.matmul(np.transpose(np.atleast_2d(xm)),np.atleast_2d(theta))
    nz=np.matmul(np.transpose(np.atleast_2d(xn)),np.atleast_2d(zeta))
    cosmt=np.cos(mt)
    sinmt=np.sin(mt)
    cosnz=np.cos(nz)
    sinnz=np.sin(nz)
    f = np.zeros((ns,lt,lz))
    fmn = np.ndarray((mn,lt))
    for k in range(ns):
        fmn = np.broadcast_to(fmnc[k,:],(lt,mn)).transpose()
        f[k,:,:]=np.matmul((fmn*sinmt).transpose(),cosnz)+np.matmul((fmn*cosmt).transpose(),sinnz)
    return f


class VMECout(SortedDict):
    """
    OMFITobject used to interact VMEC wout file

    :param filename: filename passed to OMFITnc class

    All additional key word arguments passed to OMFITnc

    """

    def __init__(self, filename, **kwargs):
        import xarray
        SortedDict.__init__(self)
        self['vmec_data'] = xarray.open_dataset(filename)
        self.dynaLoad = True

    def load(self):
        self['ns'] = int(self['vmec_data']['ns'].values)
        self['nu'] = int(self['vmec_data']['mpol'].values*4)
        self['nv'] = int(self['vmec_data']['ntor'].values*4*self['vmec_data']['nfp'].values)
        self['nv2'] = int(self['vmec_data']['ntor'].values*4)
        self['nflux'] = np.linspace(0,1,self['ns']) # np.ndarray((self['ns'],1))
        self['theta'] = np.linspace(0,2*np.pi,self['nu']) # np.ndarray((self['nu'],1))
        self['zeta'] = np.linspace(0,2*np.pi,self['nv']) # np.ndarray((self['nv'],1))
        self['zeta2']=self['zeta'][0:self['nv2']+1]
        self['r']=cfunct(self['theta'],self['zeta'],self['vmec_data']['rmnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['z']=sfunct(self['theta'],self['zeta'],self['vmec_data']['zmns'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['b']=cfunct(self['theta'],self['zeta'],self['vmec_data']['bmnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['g']=cfunct(self['theta'],self['zeta'],self['vmec_data']['gmnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['bu']=cfunct(self['theta'],self['zeta'],self['vmec_data']['bsupumnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['bv']=cfunct(self['theta'],self['zeta'],self['vmec_data']['bsupvmnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['cu']=cfunct(self['theta'],self['zeta'],self['vmec_data']['currumnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['cv']=cfunct(self['theta'],self['zeta'],self['vmec_data']['currvmnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['b_s']=sfunct(self['theta'],self['zeta'],self['vmec_data']['bsubsmns'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['b_u']=cfunct(self['theta'],self['zeta'],self['vmec_data']['bsubumnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)
        self['b_v']=cfunct(self['theta'],self['zeta'],self['vmec_data']['bsubvmnc'].values,self['vmec_data']['xm'].values,self['vmec_data']['xn'].values)

    def plot(self, plot_name='Summary', ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if (plot_name == 'Summary'):
                print(plot_name)
        elif (plot_name == 'Iota'):
                ax.plot(self['nflux'],self['vmec_data']['iotaf'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('iota')
                ax.set_title('Rotational Transform')
                #ax.set(xlabel='s',ylabel='iota',aspect='square')
        elif (plot_name == 'q'):
                ax.plot(self['nflux'],1.0/self['vmec_data']['iotaf'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('q')
                ax.set_title('Safety Factor')
        elif (plot_name == 'Pressure'):
                ax.plot(self['nflux'],self['vmec_data']['presf'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('Pressure [kPa]')
                ax.set_title('Pressure Profile')
        elif (plot_name == '<Buco>'):
                ax.plot(self['nflux'],self['vmec_data']['buco'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<B^u> [T]')
                ax.set_title('Flux surface Averaged B^u')
        elif (plot_name == '<Bvco>'):
                ax.plot(self['nflux'],self['vmec_data']['bvco'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<B^v> [T]')
                ax.set_title('Flux surface Averaged B^v')
        elif (plot_name == '<jcuru>'):
                ax.plot(self['nflux'],self['vmec_data']['jcuru'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<j^u> [kA/m^2]')
                ax.set_title('Flux surface Averaged j^u')
        elif (plot_name == '<jcurv>'):
                ax.plot(self['nflux'],self['vmec_data']['jcurv'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<j^v> [kA/m^2]')
                ax.set_title('Flux surface Averaged j^v')
        elif (plot_name == '<j.B>'):
                ax.plot(self['nflux'],self['vmec_data']['jdotb'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<j.B> [T*kA/m^2]')
                ax.set_title('Flux surface Averaged j.B')
        elif (plot_name == 'LPK'):
                ax.plot(self['r'][self['ns']-1,:,0],self['z'][self['ns']-1,:,0],color='red')
                ax.plot(self['r'][0,0,0],self['z'][0,0,0],'+',color='red')
                ax.plot(self['r'][self['ns']-1,:,int(self['nv2']/4)],self['z'][self['ns']-1,:,int(self['nv2']/4)],color='green')
                ax.plot(self['r'][0,0,int(self['nv2']/4)],self['z'][0,0,int(self['nv2']/4)],'+',color='green')
                ax.plot(self['r'][self['ns']-1,:,int(self['nv2']/2)],self['z'][self['ns']-1,:,int(self['nv2']/2)],color='blue')
                ax.plot(self['r'][0,0,int(self['nv2']/2)],self['z'][0,0,int(self['nv2']/2)],'+',color='blue')
                ax.set_xlabel('R [m]')
                ax.set_ylabel('Z [m]')
                ax.set_title('LPK Plot')
                ax.set_aspect('equal')
        elif (plot_name[0] == '-'):
                print(plot_name)
        else:
            return
