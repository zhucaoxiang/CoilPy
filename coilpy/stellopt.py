from __future__ import print_function, absolute_import
from builtins import map, filter, range
from .sortedDict import SortedDict
import numpy as np 
import matplotlib.pyplot as plt

class OMFITascii(object):
    """
    OMFIT class used to interface with ASCII files

    :param filename: filename passed to OMFITobject class

    :param fromString: string that is written to file

    :param \**kw: keyword dictionary passed to OMFITobject class
    """
    def __init__(self, filename, **kw):
        fromString = kw.pop('fromString', None)
        self.filename = filename
        if fromString is not None:
            with open(self.filename, 'wb') as f:
                f.write(fromString.encode('utf-8'))

    def read(self):
        '''
        Read ASCII file and return content

        :return: string
        '''
        return open(self.filename, 'r').read()

    def write(self, value):
        '''
        Write string value to ASCII file

        :param value: string to be written to file

        :return: string
        '''
        open(self.filename, 'w').write(value)
        return value

    def append(self, value):
        '''
        Append string value to ASCII file

        :param value: string to be written to file

        :return: string
        '''
        open(self.filename, 'a').write(value)
        return self.read()

__all__ = ['STELLout', 'VMECout', 'OMFITascii']#, 'OMFITfocuscoils']

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

class STELLout(SortedDict, OMFITascii):
    """
    OMFITobject used to interface with stellopt.* file in STELLOPT outputs.

    :param filename: filename passed to OMFITascii class

    All additional key word arguments passed to OMFITascii

    """

    def __init__(self, filename, **kwargs):
        SortedDict.__init__(self)
        OMFITascii.__init__(self, filename, **kwargs)
        self.dynaLoad = True

    def load(self):
        """Load the file and parse it into a sorted dictionary"""
        file_handle = open(self.filename,'r')
        niter = 0
        for line in file_handle:
            if 'ITER' in line:
                niter=niter+1
            if 'MIN' in line:
                niter=niter-1
        self['ITER'] = np.ndarray((niter,1));
        file_handle.seek(0)
        line = file_handle.readline()
        ttype,wh=line.split()
        self[ttype] = float(wh)

        # Enter Loop
        citer = -1
        while True:
            line = file_handle.readline()
            if line == '':
                break
            ttype,hw = line.split(' ',1)
            if ttype == 'ITER':
                if 'MIN' in hw:
                    break
                citer = citer+1
                self[ttype][citer] = int(hw)
                continue
            else:
                h,w = hw.split()
                h = int(h)
                w = int(w)
                line = file_handle.readline()
            if ttype not in self:
                self[ttype]=np.ndarray((niter,h,w))
            for i in range(h):
                line = file_handle.readline()
                val = np.fromstring(line,sep=' ')
                self[ttype][citer,i,:] = val       
        file_handle.close()
        for item in list(self):
            # print(item)
            if 'VERSION' == item:
                continue
            elif 'ITER' == item:
                continue
            elif item in ['ASPECT','ASPECT_MAX','BETA','CURTOR','PHIEDGE', \
                        'VOLUME','WP','RBTOR','R0','Z0','BETATOR','BETAPOL']:
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
            elif item == 'BALLOON':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_grate'] = np.squeeze(self[item][:,:,3])
                self[item+'_theta'] = np.squeeze(self[item][:,:,4])
                self[item+'_zeta'] = np.squeeze(self[item][:,:,5])
                self[item+'_k'] = np.squeeze(self[item][:,:,6])
            elif item == 'B_PROBES':
                self[item+'_target'] = np.squeeze(self[item][:,:,4])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,5])
                self[item+'_equil'] = np.squeeze(self[item][:,:,6])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_X'] = np.squeeze(self[item][:,:,0])
                self[item+'_Y'] = np.squeeze(self[item][:,:,1])
                self[item+'_Z'] = np.squeeze(self[item][:,:,2])
                self[item+'_MODB'] = np.squeeze(self[item][:,:,3])
            elif item in ['FLUXLOOPS','SEGROG']:
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
            elif item == 'EXTCUR':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_dex'] = np.squeeze(self[item][:,:,3])
            elif item in ['SEPARATRIX','LIMITER']:
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_R'] = np.squeeze(self[item][:,:,4])
                self[item+'_PHI'] = np.squeeze(self[item][:,:,5])
                self[item+'_Z'] = np.squeeze(self[item][:,:,6])
            elif item in ['TI','TE','IOTA','VPHI','PRESS','NE']:
                self[item+'_target'] = np.squeeze(self[item][:,:,4])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,5])
                self[item+'_equil'] = np.squeeze(self[item][:,:,6])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_R'] = np.squeeze(self[item][:,:,0])
                self[item+'_PHI'] = np.squeeze(self[item][:,:,1])
                self[item+'_Z'] = np.squeeze(self[item][:,:,2])
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
            elif item in ['NELINE','FARADAY','SXR']:
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_R0'] = np.squeeze(self[item][:,:,3])
                self[item+'_PHI0'] = np.squeeze(self[item][:,:,4])
                self[item+'_Z0'] = np.squeeze(self[item][:,:,5])
                self[item+'_R1'] = np.squeeze(self[item][:,:,6])
                self[item+'_PHI1'] = np.squeeze(self[item][:,:,7])
                self[item+'_Z1'] = np.squeeze(self[item][:,:,8])
            elif item == 'MSE':
                self[item+'_target'] = np.squeeze(self[item][:,:,4])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,5])
                self[item+'_equil'] = np.squeeze(self[item][:,:,8])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_R'] = np.squeeze(self[item][:,:,0])
                self[item+'_PHI'] = np.squeeze(self[item][:,:,1])
                self[item+'_Z'] = np.squeeze(self[item][:,:,2])
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
                self[item+'_ER'] = np.squeeze(self[item][:,:,6])
                self[item+'_EZ'] = np.squeeze(self[item][:,:,7])
            elif item == 'BOOTSTRAP':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
                self[item+'_avg_jdotb'] = np.squeeze(self[item][:,:,4])
                self[item+'_beam_jdotb'] = np.squeeze(self[item][:,:,5])
                self[item+'_boot_jdotb'] = np.squeeze(self[item][:,:,6])
                self[item+'_jBbs'] = np.squeeze(self[item][:,:,7])
                self[item+'_facnu'] = np.squeeze(self[item][:,:,8])
                self[item+'_bsnorm'] = np.squeeze(self[item][:,:,9])
            elif item == 'HELICITY':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_bnorm'] = np.squeeze(self[item][:,:,3])
                self[item+'_m'] = np.squeeze(self[item][:,:,4])
                self[item+'_n'] = np.squeeze(self[item][:,:,5])
            elif item ==  'HELICITY_FULL':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_bnorm'] = np.squeeze(self[item][:,:,3])
                self[item+'_k'] = np.squeeze(self[item][:,:,4])
                self[item+'_m'] = np.squeeze(self[item][:,:,5])
                self[item+'_n'] = np.squeeze(self[item][:,:,6])               
            elif item == 'TXPORT':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
            elif item == 'COIL_BNORM':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_U'] = np.squeeze(self[item][:,:,3])
                self[item+'_V'] = np.squeeze(self[item][:,:,4])
                self[item+'_BNEQ'] = np.squeeze(self[item][:,:,5])
                self[item+'_BNF'] = np.squeeze(self[item][:,:,6])
            elif item == 'ORBIT':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
            elif item == 'J_STAR':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_AVGJSTAR'] = np.squeeze(self[item][:,:,3])
                self[item+'_TRAPSJSTAR'] = np.squeeze(self[item][:,:,4])
                self[item+'_UJSTAR'] = np.squeeze(self[item][:,:,5])
                self[item+'_K'] = np.squeeze(self[item][:,:,6])
                self[item+'_IJSTAR'] = np.squeeze(self[item][:,:,7])
            elif item == 'NEO':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_K'] = np.squeeze(self[item][:,:,3])
            elif item == 'JDOTB':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
            elif item == 'JTOR':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
            elif item == 'DKES':
                self[item+'_target'] = np.squeeze(self[item][:,:,0])
                self[item+'_sigma'] = np.squeeze(self[item][:,:,1])
                self[item+'_equil'] = np.squeeze(self[item][:,:,2])
                self[item+'_chisq'] = ((self[item+'_target'] - self[item+'_equil'])/self[item+'_sigma'])**2
                self[item+'_S'] = np.squeeze(self[item][:,:,3])
                self[item+'_NU'] = np.squeeze(self[item][:,:,4])
                self[item+'_ER'] = np.squeeze(self[item][:,:,5])
                self[item+'_L11P'] = np.squeeze(self[item][:,:,6])
                self[item+'_L11M'] = np.squeeze(self[item][:,:,7])
                self[item+'_L33P'] = np.squeeze(self[item][:,:,8])
                self[item+'_L33M'] = np.squeeze(self[item][:,:,9])
                self[item+'_L31P'] = np.squeeze(self[item][:,:,10])
                self[item+'_L31M'] = np.squeeze(self[item][:,:,11])
                self[item+'_SCAL11'] = np.squeeze(self[item][:,:,12])
                self[item+'_SCAL33'] = np.squeeze(self[item][:,:,13])
                self[item+'_SCAL31'] = np.squeeze(self[item][:,:,14])
    def plot_helicity(self, it=-1, ordering=0, mn=(None, None), ax=None, **kwargs):     
        """Plot |B| components in Boozer coordinates from BOOZ_XFORM

        Args:
            it (int, optional): Iteration index to be plotted. Defaults to -1.
            ordering (integer, optional): Plot the leading Nordering asymmetric modes. Defaults to 0.
            mn (tuple, optional): Plot the particular (m,n) mode. Defaults to (None, None).
            ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
        
        Returns:
            data (numpy.ndarray): The selected Fourier harmonics.
        """        
        xs = self['HELICITY_FULL_k']
        xs = np.array(np.unique(xs), dtype=int)
        ns = len(xs)
        xx = (xs-1)/np.max(xs) # max(xs) might be different from NS
        vals = np.reshape(self['HELICITY_FULL_equil'][it], (ns, -1))
        xm = np.reshape(np.array(self['HELICITY_FULL_m'][it], dtype=int), (ns, -1))
        xn = np.reshape(np.array(self['HELICITY_FULL_n'][it], dtype=int), (ns, -1))
        # get figure and ax data
        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        else :
            fig, ax = plt.subplots()
        if ordering:
            assert ordering >= 1
            data = np.linalg.norm(vals, axis=0)
            ind_arg = np.argsort(data)
            for i in range(ordering):
                ind = ind_arg[-1-i] # index of the i-th largest term
                m = xm[0, ind]
                n = xn[0, ind]
                kwargs['label'] = 'm={:}, n={:}'.format(m,n)
                ax.plot(xx, vals[:, ind], **kwargs)
            ylabel = r'$\frac{B_{m,n}}{ \Vert B_{n=0} \Vert }$'
        else:
            # determine filter condition
            if mn[0] is not None:
                mfilter = (xm == mn[0])
                m = 'm={:}'.format(mn[0])
            else:
                mfilter = np.full(np.shape(xm), True)
                m = 'm'
            if mn[1] is not None:
                nfilter = (xn == mn[1])
                n = 'n={:}'.format(mn[1])
            else:
                nfilter = (xn != 0)
                n = r'n \neq 0'
            cond = np.logical_and(mfilter, nfilter)
            data = np.reshape(vals[cond], (ns, -1))
            line = ax.plot(xx, np.linalg.norm(data, axis=1), **kwargs)
            ylabel = r'$ \frac{{ \Vert B_{{ {:},{:} }} \Vert }}{{ \Vert B_{{n=0}} \Vert }} $'.format(m, n)
        plt.xlabel('normalized flux (s)', fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        return data

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
