# This script contains some example of preprocessing an dataset into a hdf5 file that can be read into the
# PtyLab dataset.
import numpy as np
import matplotlib.pylab as plt
import imageio
import tqdm
from skimage.transform import rescale
import glob
import os
import h5py


filePathForRead = '/run/user/1001/gvfs/smb-share:server=filer.uni-siegen.de,share=xnas/CSE/CSE-Research/Microscopy3D/CV_CSE_Collaboration/Data/Tian14_ResTarget/1LED/'
#filePathForRead = '/run/user/1001/gvfs/smb-share:server=filer.uni-siegen.de,share=xnas/CSE/CSE-Research/Microscopy3D/CV_CSE_Collaboration/Data/Tian14_ResTarget/tmp/'
filePathForSave = '/run/user/1001/gvfs/smb-share:server=filer.uni-siegen.de,share=xnas/CSE/CSE-Research/Microscopy3D/CV_CSE_Collaboration/Results/CV_CSE/PtyLab/'
# D:\Du\Workshop\fracmat\lenspaper4\AVT camera (GX1920)
# D:/fracmat/ptyLab/lenspaper4/AVT camera (GX1920)
os.chdir(filePathForRead)

#fileName = "ResTarget"
fileName = "ResTarget_v6"
# wavelength
wavelength = 6292e-9
# binning
binningFactor = 1
# padding for superresolution
padFactor = 1
# set magnification if any objective lens is used
magnification = 10
# LED object distance
zled = 67.5e-3
# object detector distance  (initial guess)
zo = zled*3
# Numerical Aperture
NA = 0.3


# set detection geometry
# A: camera to closer side of stage (allows to bring camera close in transmission)
# B: camera to further side of stage (doesn't allow to bring close in transmission),
# or other way around in reflection
# C: objective + tube lens in transmission
measurementMode = "A"
# camera
camera = "TianResTarget"
#camera = "GX"
if camera == "GX":
    N = 1456
    M = 1456
    dxd = (
        4.54e-6 * binningFactor / magnification
    )  # effective detector pixel size is magnified by binning
    backgroundOffset = 100  # globally subtracted from raw data (diffraction intensities), play with this value
elif camera == "Hamamatsu":
    N = 2**11
    M = 2**11
    dxd = 6.5e-6 * binningFactor / magnification
    backgroundOffset = 0
elif camera == "TianResTarget":
    N = 2160
    #M = 2560
    M = 2160
    dxd = (
       6.5e-6 * binningFactor / magnification
    )  # effective detector pixel size is magnified by binning
    backgroundOffset = 100  # globally subtracted from raw data (diffraction intensities), play with this value    

# number of frames is calculated automatically
framesList = glob.glob("*" + ".tif")
framesList.sort()
#numFrames = len(framesList) - 1
numFrames = len(framesList)

print(f'framesList.shape:{framesList} ')

# read background
#dark = imageio.imread("background.tif")
#dark = np.zeros((N, M)).astype("float32")

# read empty beam (if available)

# binning
ptychogram = np.zeros(
    (numFrames, N // binningFactor * padFactor, M // binningFactor * padFactor),
    dtype=np.float32,
)

# read frames
pbar = tqdm.trange(numFrames, leave=True)
for k in pbar:
    # get file name
    pbar.set_description("reading frame" + framesList[k])
    temp = imageio.imread(framesList[k]).astype("float32") 
    temp = temp[:,0:2160] 
    print(f'temp.shape:{temp.shape} ')
    #print(f'dark.shape:{dark.shape} ')
    #temp = temp - dark - backgroundOffset
    temp[temp < 0] = 0  # todo check if data type is single
    # crop
    #temp = temp[
    #    M // 2 - N // 2 : M // 2 + N // 2 - 1, M // 2 - N // 2 : M // 2 + N // 2 - 1
    #]
    # binning
    temp = rescale(
        temp, 1 / binningFactor, order=0
    )  # order = 0 takes the nearest-neighbor
    # flipping
    if measurementMode == "A":
        temp = np.flipud(temp)
    elif measurementMode == "B":
        temp = np.rot90(temp, axes=(0, 1))
    elif measurementMode == "C":
        temp = np.rot90(np.flipud(temp), axes=(0, 1))

    # zero padding
    ptychogram[k] = np.pad(temp, (padFactor - 1) * N // binningFactor // 2)

# set experimental specifications:
entrancePupilDiameter = 1000e-6

# object coordinates
#No = 2**11 + 2**9 #TODO: this needs to be fixed! -> this is probably the dimesnsions of the high resolution image
No = 2160 #TODO: this needs to be fixed! -> this is probably the dimesnsions of the high resolution image

# detector coordinates
Nd = ptychogram.shape[-1]  # number of detector pixels
Ld = Nd * dxd  # effective size of detector
xd = np.arange(-Nd // 2, Nd // 2) * dxd  # 1D coordinates in detector plane
Xd, Yd = np.meshgrid(xd, xd)  # 2D coordinates in detector plane

# get positions
# get file name (this assumes there is only one text file in the raw data folder)
positionFileName = glob.glob("dirac_positions" + ".txt")[0]
idx_led = glob.glob("idx_led" + ".txt")[0]

# Load raw data positions
T = np.genfromtxt(positionFileName, delimiter="\t", skip_header=0)

# Load idx_led values
idx_led_values = np.genfromtxt(idx_led, dtype=int)

# Sort T array based on idx_led values
sorted_indices = np.argsort(idx_led_values)
T_sorted = T[sorted_indices]

# Convert to micrometer
encoder = (T_sorted - T_sorted[0]) * 1e-6



print(f'encoder.shape:{encoder.shape} ')

# show positions
plt.figure(figsize=(5, 5))
plt.plot(encoder[:, 1] * 1e6, encoder[:, 0] * 1e6, "o-")
plt.xlabel("(um))")
plt.ylabel("(um))")
plt.show()

# export data
exportBool = True

if exportBool:
    os.chdir(filePathForSave)
    hf = h5py.File(fileName + ".hdf5", "w")
    hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
    hf.create_dataset("encoder", data=encoder, dtype="f")
    hf.create_dataset("binningFactor", data=(binningFactor,), dtype="i")
    hf.create_dataset("dxd", data=(dxd,), dtype="f")
    hf.create_dataset("Nd", data=(Nd,), dtype="i")
    hf.create_dataset("No", data=(No,), dtype="i")
    hf.create_dataset("zo", data=(zo,), dtype="f")
    hf.create_dataset("zled", data=(zled,), dtype="f")
    hf.create_dataset("magnification", data=(magnification,), dtype="f")
    hf.create_dataset("wavelength", data=(wavelength,), dtype="f")
    hf.create_dataset("NA", data=(NA,), dtype="f")
    hf.create_dataset("entrancePupilDiameter", data=(entrancePupilDiameter,), dtype="f")
    hf.close()
    print("An hd5f file has been saved")
