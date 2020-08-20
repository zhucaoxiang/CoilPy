# modified from stellopt.pySTEL

import numpy as np 
import xarray
import matplotlib.pyplot as plt
from .misc import trig2real
from .surface import FourSurf

__all__ = ['VMECout']

class VMECout(object):
    """
    VMEC wout file

    :param filename: filename passed to OMFITnc class

    All additional key word arguments passed to OMFITnc

    """

    def __init__(self, filename, **kwargs):
        self.wout = xarray.open_dataset(filename)
        self.data = {}
        self.data['ns'] = int(self.wout['ns'].values)
        self.data['nfp'] = int(self.wout['nfp'].values)
        self.data['nu'] = int(self.wout['mpol'].values*4)
        self.data['nv'] = int(self.wout['ntor'].values*4*self.wout['nfp'].values)
        self.data['nv2'] = int(self.wout['ntor'].values*4)
        self.data['nflux'] = np.linspace(0,1,self.data['ns']) # np.ndarray((self.data['ns'],1))
        self.data['theta'] = np.linspace(0,2*np.pi,self.data['nu']) # np.ndarray((self.data['nu'],1))
        self.data['zeta'] = np.linspace(0,2*np.pi,self.data['nv']) # np.ndarray((self.data['nv'],1))
        self.data['zeta2']= self.data['zeta'][0:self.data['nv2']+1]
        self.surface = []
        self.data['b'] = []
        for i in range(self.data['ns']):
            self.surface.append(FourSurf(xm=self.wout['xm'].values, xn=self.wout['xn'].values,
                                         rbc=self.wout['rmnc'][i].values, rbs=np.zeros_like(self.wout['rmnc'][i].values),
                                         zbs=self.wout['zmns'][i].values, zbc=np.zeros_like(self.wout['zmns'][i].values)))
            self.data['b'].append(trig2real(self.data['theta'], self.data['zeta'],
                                            self.wout['xm_nyq'].values, self.wout['xn_nyq'].values/self.data['nfp'],
                                            self.wout['bmnc'][i].values))
        return

    def plot(self, plot_name='none', ax=None):
        """Plot various VMEC quantities

        Args:
            plot_name (str, optional): The quantity to be plotted, should be one of
                                       iota, q, pressue, <Buco>, <Bvco>, <jcuru>, <jcurv>,
                                       <j.B>, LPK, none. Defaults to 'none'.
            ax (Matplotlib axis, optional): The Matplotlib axis to be plotted on. Defaults to None.
        """        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if (plot_name == 'none'):
                print("You can plot: iota, q, pressue, <Buco>, <Bvco>, <jcuru>, <jcurv>, ")
                Print("               <j.B>, LPK")
        elif (plot_name == 'iota'):
                ax.plot(self.data['nflux'],self.wout['iotaf'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('iota')
                ax.set_title('Rotational Transform')
                #ax.set(xlabel='s',ylabel='iota',aspect='square')
        elif (plot_name == 'q'):
                ax.plot(self.data['nflux'],1.0/self.wout['iotaf'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('q')
                ax.set_title('Safety Factor')
        elif (plot_name == 'pressure'):
                ax.plot(self.data['nflux'],self.wout['presf'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('Pressure [kPa]')
                ax.set_title('Pressure Profile')
        elif (plot_name == '<Buco>'):
                ax.plot(self.data['nflux'],self.wout['buco'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<B^u> [T]')
                ax.set_title('Flux surface Averaged B^u')
        elif (plot_name == '<Bvco>'):
                ax.plot(self.data['nflux'],self.wout['bvco'].values)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<B^v> [T]')
                ax.set_title('Flux surface Averaged B^v')
        elif (plot_name == '<jcuru>'):
                ax.plot(self.data['nflux'],self.wout['jcuru'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<j^u> [kA/m^2]')
                ax.set_title('Flux surface Averaged j^u')
        elif (plot_name == '<jcurv>'):
                ax.plot(self.data['nflux'],self.wout['jcurv'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<j^v> [kA/m^2]')
                ax.set_title('Flux surface Averaged j^v')
        elif (plot_name == '<j.B>'):
                ax.plot(self.data['nflux'],self.wout['jdotb'].values/1000)
                ax.set_xlabel('Normalized Flux')
                ax.set_ylabel('<j.B> [T*kA/m^2]')
                ax.set_title('Flux surface Averaged j.B')
        elif (plot_name == 'LPK'):
                self.surface[-1].plot(zeta=0, color='red', label=r'$\phi=0$')
                self.surface[-1].plot(zeta=0.5*np.pi/self.data['nfp'], color='green', label=r'$\phi=0.25$')
                self.surface[-1].plot(zeta=np.pi/self.data['nfp'], color='blue', label=r'$\phi=0.5$')
                ax.set_title('LPK Plot')
        elif (plot_name[0] == '-'):
                print(plot_name)
        else:
            return
