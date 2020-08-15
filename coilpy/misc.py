#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys

def xy2rp(x, y):
    """Convert (x,y) to (R,phi) in polar coordinate

    Args:
        x (float): x coordinate
        y (float): y coordinate

    Returns:
        R (float): radius
        phi (float): angle in rad
    """    
    R = np.sqrt(x**2 + y**2)
    if   x >  0.0 and y >= 0.0 : # [0,pi/2)
        phi = np.arcsin(y/R)
    elif x <= 0.0 and y >  0.0 : # [pi/2, pi)
        phi = np.arccos(x/R)
    elif x <  0.0 and y <= 0.0 : # [pi, 3/2 pi)
        phi = np.arccos(-x/R) + np.pi
    elif x >= 0.0 and y <  0.0 : # [3/2 pi, 2pi)
        phi = np.arcsin(y/R) + 2*np.pi
    else:
        raise ValueError("Something wrong with your inputs ({:f}, {:f}).".format(x, y))
    return R, phi

def map_matrix(xx, first=True, second=True):
    """Map matrix to be complete (closed)
    Arguments:
      xx -- 2D numpy array
      first -- boolean, default: True, if increase the first dimension
      second -- boolean, default: True, if increase the second dimension

    Returns:
      new -- the new matrix with dimension increased
    """
    a, b = np.shape(xx)
    # only first
    if first and not second :
        new = np.zeros((a+1,b))
        new[0:a, 0:b] = xx[0:a, 0:b]
        new[  a, 0:b] = xx[0  , 0:b]
    # only second
    elif not first and second :
        new = np.zeros((a,b+1))
        new[0:a, 0:b] = xx[0:a, 0:b]
        new[0:a,   b] = xx[0:a, 0  ]
    # both direction
    elif first and second :
        new = np.zeros((a+1,b+1))
        new[0:a, 0:b] = xx[0:a, 0:b]
        new[  a, 0:b] = xx[0  , 0:b]
        new[0:a,   b] = xx[0:a, 0  ]
        new[  a,   b] = xx[0  , 0  ]
    # otherwise return the original matrix
    else :
        return xx
    return new

def toroidal_period(vec, nfp=1):
    """
    vec: [x,y,z] data
    Nfp: =1, toroidal number of periodicity
    """
    phi = 2*np.pi/nfp
    vec = np.atleast_2d(vec)
    new_vec = vec.copy()
    for ifp in range(nfp):
        if ifp==0:
            continue
        rotate = np.array([[  np.cos(ifp*phi), np.sin(ifp*phi), 0], \
                           [ -np.sin(ifp*phi), np.cos(ifp*phi), 0], \
                           [                0,               0, 1]])
        new_vec = np.concatenate((new_vec, np.matmul(vec, rotate)))
    return new_vec

def print_progress(iteration, total, prefix='Progress', suffix='Complete', decimals=1, bar_length=60):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    return

# Smart way to check where to plot
def get_figure(axes=None, **kwargs):
    """
    Check where to plot
    Parameters:
        axes: matplotlib.pyplot axis, axis to plot (default: None)
        kwargs: keyword arguments
    Return:
        f, ax 
        f  : matplotlib.pyplot figure
        ax : matplotlib.pyplot axis 
    """
    import matplotlib.pyplot as plt
    if axes is None:
        # No axes provided
        f, axes = plt.subplots(**kwargs)
        '''
        f = plt.gcf()
        if len(f.axes):
            # normal situation in which existing figures should be respected and left alone
            f, axes = plt.subplots(**kwargs)
        else:
            #  made a empty figure for using
            axes = f.add_subplot(**kwargs)
         '''
    else:
        # axes = np.atleast_1d(axes)
        f = axes.get_figure()
    return f, axes

def colorbar(mappable, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)
    plt.sca(last_axes)
    return cbar

def kwargs2dict(**kwargs):
    return kwargs

def vmecMN(mpol, ntor):
# manipulate VMEC index
    mn = (2*ntor+1)*mpol - ntor # total number of Fourier harmonics
    xm = np.zeros((mn,), dtype=int)
    xn = np.zeros((mn,), dtype=int)
    imn = 0 
    for ii in range(mpol):
        for jj in range(-ntor, ntor+1):
            if ii==0 and jj<0 :
                continue
            xm[imn] = ii
            xn[imn] = jj
            imn += 1
    return xm, xn

def trigfft(y, tr=-1):
    """calculate trigonometric coefficients using FFT
    Assuming the periodicity is 2*pi
    params:
        y -- 1D array for Fourier transformation
        tr -- Truncation number (default: -1)
    return:
        a dict containing
        'n' -- index
        'rcos' -- cos coefficients of the real part
        'rsin' -- sin coefficients of the real part
        'icos' -- cos coefficients of the imag part
        'isin' -- sin coefficients of the imag part
    """
    from scipy.fftpack import fft
    N = len(y)
    if N%2 == 0 : # even
        half = N//2-1
        end = half + 2
    else :
        half = (N-1)//2
        end = half + 1
    assert tr <= end, 'Truncation number should be smaller than dimension!'
    comp = fft(y)/N
    a_k = np.zeros(end, dtype=np.complex)
    b_k = np.zeros(end, dtype=np.complex)
    a_k[0] = comp[0]
    for n in range(1, half+1):
        a_k[n] =  comp[n] + comp[N-n]
        b_k[n] = (comp[n] - comp[N-n]) * 1j
    if N%2 == 0: # even number
        a_k[end-1] = comp[N//2]
    index = np.arange(end)
    
    return {'n': index[:tr],
            'rcos': np.real(a_k[:tr]),
            'rsin': np.real(b_k[:tr]), 
            'icos': np.imag(a_k[:tr]), 
            'isin': np.imag(b_k[:tr])
           }

def fft_deriv(y):
    from scipy.fftpack import fft, ifft
    N = len(y)
    comp = fft(y)
    if N%2 == 0:
        dt = (np.arange(N) - np.concatenate((np.zeros(N//2), [N//2], N*np.ones(N//2-1))))*1j
    else:
        dt = (np.arange(N) - np.concatenate((np.zeros(N//2), N*np.ones(N//2+1))))*1j
    return ifft(comp*dt)

def trig2real(theta, zeta, xm, xn, fmnc=None, fmns=None):
        npol, ntor = len(theta), len(zeta)
        _tv, _zv = np.meshgrid(theta, zeta, indexing='ij')
        # mt - nz (in matrix)
        _mtnz = np.matmul( np.reshape(xm, (-1,1)), np.reshape(_tv, (1,-1)) ) \
              - np.matmul( np.reshape(xn, (-1,1)), np.reshape(_zv, (1,-1)) ) 
        _cos = np.cos(_mtnz)
        _sin = np.sin(_mtnz)

        f = np.zeros((1, npol*ntor))
        if fmnc is not None:
            f += np.matmul(np.reshape(fmnc, (1,-1)), _cos)
        if fmns is not None:
            f += np.matmul(np.reshape(fmns, (1,-1)), _sin)
        return f.reshape(npol, ntor)     
