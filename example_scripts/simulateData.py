# This script contains a minimum working example of how to generate data
import numpy as np
from fracPy.utils.utils import circ, gaussian2D, cart2pol
from fracPy.utils.scanGrids import GenerateNonUniformFermat
from fracPy.operators.operators import aspw
from fracPy.utils.visualisation import hsvplot
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.monitors.Monitor import Monitor
import imageio
import tqdm
from skimage.transform import rescale
import glob
import os
import h5py

fileName = 'simuRecent'
# create ptyLab object
simuData = ExperimentalData()

# set physical properties

simuData.wavelength = 632.8e-9
simuData.zo = 5e-2
simuData.nlambda = 1
simuData.nosm = 1
simuData.npsm = 1
simuData.nslice = 1


# detector coordinates
simuData.Nd = 2**7
simuData.dxd = 2**11/simuData.Nd*4.5e-6
# simuData.Ld = simuData.Nd*simuData.dxd
# simuData.xd = np.arange(-simuData.Nd//2, simuData.Nd//2) * simuData.dxd
# simuData.Xd, simuData.Yd = np.meshgrid(simuData.xd, simuData.xd)          # 2D coordinates in detector plane

# probe coordinates
# simuData.dxp = simuData.wavelength * simuData.zo / simuData.Ld
simuData.Np = simuData.Nd
# simuData.Lp = simuData.Np * simuData.dxp
# simuData.xp = np.arange(-simuData.Np//2, simuData.Np//2) * simuData.dxp
# simuData.Xp, simuData.Yp = np.meshgrid(simuData.xp, simuData.xp)
simuData.zp = 1e-2  # pinhole-object distance

# object coordinates
# simuData.dxo = simuData.dxp
simuData.No = 2**10
# simuData.Lo = simuData.No * simuData.dxo
# simuData.xo = np.arange(-simuData.No//2, simuData.No//2) * simuData.dxo
# simuData.Xo, simuData.Yo = np.meshgrid(simuData.xo, simuData.xo)

# generate illumination
# note: simulate focused beam
# goal: 1:1 image iris through (low-NA) lens with focal length f onto an object
f = 5e-3 # focal length of lens, creating a focused probe
pinhole = circ(simuData.Xp, simuData.Yp, simuData.Lp/2)
pinhole = convolve2d(pinhole, gaussian2D(5, 1), mode='same')

# propagate to lens
probe = aspw(pinhole, 2*f, simuData.wavelength, simuData.Lp)[0]

# multiply with quadratic phase and aperture
aperture = circ(simuData.Xp, simuData.Yp, 3*simuData.Lp/4)
aperture = convolve2d(aperture, gaussian2D(5, 3), mode='same')
probe = probe * np.exp(-1.j*2*np.pi/simuData.wavelength*(simuData.Xp**2+simuData.Yp**2)/(2*f)) * aperture
probe = aspw(probe, 2*f, simuData.wavelength, simuData.Lp)[0]

simuData.probe = np.zeros((simuData.nlambda, 1, simuData.npsm, simuData.nslice, simuData.Np, simuData.Np), dtype='complex64')
simuData.probe[...,:,:] = probe

plt.figure(figsize=(5,5), num=1)
ax1 = plt.subplot(121)
hsvplot(probe, ax=ax1, pixelSize=simuData.dxp)
ax1.set_title('complex probe')
plt.subplot(122)
plt.imshow(abs(probe)**2)
plt.title('probe intensity')
plt.show(block=False)

# generate object
d = 1e-3   # the smaller this parameter the larger the spatial frequencies in the simulated object
b = 33     # topological charge (feel free to play with this number)
theta, rho = cart2pol(simuData.Xo, simuData.Yo)
t = (1 + np.sign(np.sin(b * theta + 2*np.pi * (rho/d)**2)))/2
# phaseFun = np.exp(1.j * np.atan2(Yo, Xo))
phaseFun = 1
# phaseFun = np.exp(1.j*( 1 * theta + 2*np.pi * (rho/d)**2))
t = t*circ(simuData.Xo, simuData.Yo, simuData.Lo)*(1-circ(simuData.Xo, simuData.Yo, 200*simuData.dxo))*phaseFun\
    +circ(simuData.Xo, simuData.Yo, 130*simuData.dxo)
obj = convolve2d(t, gaussian2D(5, 3), mode='same')  # smooth edges
simuData.object = np.zeros((simuData.nlambda, simuData.nosm, 1, simuData.nslice, simuData.No, simuData.No), dtype='complex64')
simuData.object[...,:,:] = obj
plt.figure(figsize=(5,5), num=2)
ax = plt.axes()
hsvplot(obj, ax=ax, pixelSize=simuData.dxo)
ax.set_title('complex probe')
plt.show(block=False)

# generate positions
# generate non-uniform Fermat grid
# parameters
numPoints = 100   # number of points
radius = 1e-4    # radius of final scan grid (in meters)
p = 1    # p = 1 is standard Fermat;  p > 1 yields more points towards the center of grid
R, C = GenerateNonUniformFermat(numPoints, radius=radius, power=p)
# show scan grid
plt.figure(figsize=(5, 5), num=99)
plt.plot(1e6*R, 1e6*C, 'o')
plt.xlabel('um')
plt.title('scan grid')
plt.show(block=False)

# optimize scan grid
# numIterations = 5e4   # number of iterations in optimization
# print('optimize scan grid')
simuData.encoder = np.vstack((R, C)).T

# prevent negative indices by centering spiral coordinates on object
positions = np.round(simuData.encoder/simuData.dxo)
offset = 100
positions = (positions+simuData.No//2-simuData.Np//2+offset).astype(int)

# get number of positions
numFrames = len(R)
print('generate positions('+str(numFrames)+')')

# calculate estimated overlap
# expected beam size, required to calculate overlap (expect Gaussian-like beam, derive from second moment)
beamSize = np.sqrt(np.sum((simuData.Xp**2+simuData.Yp**2)*np.abs(simuData.probe)**2)/np.sum(abs(simuData.probe)**2))*2.355
distances = np.sqrt(np.diff(R)**2+np.diff(C)**2)
averageDistance = np.mean(distances)
print('average step size:%.1f (um)' % averageDistance)
print('number of scan points: %d' % numFrames)

# generate ptychogram
simuData.ptychogram = np.zeros((numFrames, simuData.Nd, simuData.Nd))

optimizable = Optimizable(simuData)
monitor = Monitor()
reconstructor = BaseReconstructor(optimizable, simuData, monitor)

reconstructor.propagator = 'Fresnel'
reconstructor._initializeParams()  # to calculate the quadratic phase
reconstructor.fftshiftSwitch = False
for loop in np.arange(simuData.numFrames):
    # get object patch
    row, col = positions[loop]
    sy = slice(row, row + simuData.Np)
    sx = slice(col, col + simuData.Np)
    # note that object patch has size of probe array
    objectPatch = simuData.object[..., sy, sx].copy()
    # multiply each probe mode with object patch
    optimizable.esw = objectPatch*optimizable.probe
    # generate diffraction data
    reconstructor.object2detector()
    # save data in ptychogram
    simuData.ptychogram[loop] = np.sum(abs(optimizable.ESW)**2, axis=(0, 1, 2, 3))
    # inspect diffraction data

# calcuate noise
# simulate Possion noise
bitDepth = 12
maxNumCountsPerDiff = 2**bitDepth

# normalize data (ptychogram)
# I = I/max(obj.ptychogram(:)) * maxNumCountsPerDiff;
simuData.ptychogram = simuData.ptychogram/np.max(simuData.ptychogram) * maxNumCountsPerDiff

# simulate Poisson noise
# % obj.ptychogram = poisson_noise(obj.ptychogram); % beware: slow, but requires no toolbox license
# obj.ptychogram = poisson_noise(obj.ptychogram(:)); % beware: slow, but requires no toolbox license

# todo: compare noiseless data noisy
# figure(12)
# imagesc(sqrt([I, obj.ptychogram(:,:,loop)]))
# axis image off; colormap(cmap)
# title(['left: noiseless, right: noisy (',num2str(bitDepth),' bit)'])

# set data
simuData.entrancePupilDiameter = beamSize

# todo data inspection, check sampling requirements

# export data
exportBool = True
saveFilePath = r"D:\Du\Workshop\fracpy\example_data"
os.chdir(saveFilePath)
if exportBool:
    hf = h5py.File(fileName+'.hdf5', 'w')
    hf.create_dataset('ptychogram', data=simuData.ptychogram, dtype='f')
    hf.create_dataset('encoder', data=simuData.encoder, dtype='f')
    # hf.create_dataset('positions', data=simuData.positions, dtype='f')
    hf.create_dataset('dxd', data=(simuData.dxd,), dtype='f')
    hf.create_dataset('Nd', data=(simuData.Nd,), dtype='i')
    hf.create_dataset('No', data=(simuData.No,), dtype='i')
    hf.create_dataset('Np', data=(simuData.Np,), dtype='i')
    hf.create_dataset('zo', data=(simuData.zo,), dtype='f')
    hf.create_dataset('wavelength', data=(simuData.wavelength,), dtype='f')
    hf.create_dataset('entrancePupilDiameter', data=(simuData.entrancePupilDiameter,), dtype='f')
    hf.close()
    print('An hd5f file has been saved')