import numpy as np
from matplotlib import pyplot as plt
try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

# fracPy imports
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor
from fracPy.utils.utils import fft2c, ifft2c
import logging
import tqdm
import sys


class mPIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData,monitor)
        self.logger = logging.getLogger('mPIE')
        self.logger.info('Sucesfully created mPIE mPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        # initialize mPIE params
        self.initializeReconstructionParams()
        # initialize momentum
        self.optimizable.initializeObjectMomentum()
        self.optimizable.initializeProbeMomentum()
        # set object and probe buffers
        self.optimizable.objectBuffer = self.optimizable.object.copy()
        self.optimizable.probeBuffer = self.optimizable.probe.copy()

        self.momentumAcceleration = True
        
    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the mPIE settings.
        :return:
        """
        # self.eswUpdate = self.optimizable.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1     # probe regularization
        self.alphaObject = 0.1    # object regularization
        self.feedbackM = 0.3          # feedback
        self.frictionM = 0.7          # friction
        self.probeWindow = np.abs(self.optimizable.probe)
        
    def _prepare_doReconstruction(self):
        """
        This function is called just before the reconstructions start.

        Can be used to (for instance) transfer data to the GPU at the last moment.
        :return:
        """
        pass

    def doReconstruction(self):
        self._initializeParams()
        self._prepare_doReconstruction()

        # actual reconstruction MPIE_engine
        self.pbar = tqdm.trange(self.numIterations, desc='mPIE', file=sys.stdout, leave=True)
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.experimentalData.positions[positionIndex]
                sy = slice(row, row + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[..., sy, sx].copy()
                
                # make exit surface wave
                self.optimizable.esw = objectPatch * self.optimizable.probe
                
                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                # object update
                self.optimizable.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.optimizable.probe = self.probeUpdate(objectPatch, DELTA)

                # momentum updates
                if np.random.rand(1) > 0.95:
                    self.objectMomentumUpdate()
                    self.probeMomentumUpdate()

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

            #todo clearMemory implementation

    def objectMomentumUpdate(self):
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
        """
        gradient = self.optimizable.objectBuffer - self.optimizable.object
        self.optimizable.objectMomentum = gradient + self.frictionM * self.optimizable.objectMomentum
        self.optimizable.object = self.optimizable.object - self.feedbackM * self.optimizable.objectMomentum
        self.optimizable.objectBuffer = self.optimizable.object.copy()


    def probeMomentumUpdate(self):
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
        """
        gradient = self.optimizable.probeBuffer - self.optimizable.probe
        self.optimizable.probeMomentum = gradient + self.frictionM * self.optimizable.probeMomentum
        self.optimizable.probe = self.optimizable.probe - self.feedbackM * self.optimizable.probeMomentum
        self.optimizable.probeBuffer = self.optimizable.probe.copy()


    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absP2 = xp.abs(self.optimizable.probe)**2
        Pmax = xp.max(xp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        if self.experimentalData.operationMode =='FPM':
            frac = abs(self.optimizable.probe)/Pmax*\
                   self.optimizable.probe.conj()/(self.alphaObject*Pmax+(1-self.alphaObject)*absP2)
        else:
            frac = self.optimizable.probe.conj()/(self.alphaObject*Pmax+(1-self.alphaObject)*absP2)
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=2, keepdims=True)

       
    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absO2 = xp.abs(objectPatch) ** 2
        Omax = xp.max(xp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        frac = objectPatch.conj() / (self.alphaProbe * Omax + (1-self.alphaProbe) * absO2)
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=1, keepdims=True)
        return r


class mPIE_GPU(mPIE):
    """
    GPU-based implementation of mPIE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cp is None:
            raise ImportError('Could not import cupy')
        self.logger = logging.getLogger('mPIE_GPU')
        self.logger.info('Hello from mPIE_GPU')

    def _prepare_doReconstruction(self):
        self.logger.info('Ready to start transferring stuff to the GPU')
        self._move_data_to_gpu()

    def _move_data_to_gpu(self):
        """
        Move the data to the GPU
        :return:
        """
        # optimizable parameters
        self.optimizable.probe = cp.array(self.optimizable.probe, cp.complex64)
        self.optimizable.object = cp.array(self.optimizable.object, cp.complex64)
        self.optimizable.probeBuffer = cp.array(self.optimizable.probeBuffer, cp.complex64)
        self.optimizable.objectBuffer = cp.array(self.optimizable.objectBuffer, cp.complex64)
        self.optimizable.probeMomentum = cp.array(self.optimizable.probeMomentum, cp.complex64)
        self.optimizable.objectMomentum = cp.array(self.optimizable.objectMomentum, cp.complex64)

        # non-optimizable parameters
        self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)
        # self.experimentalData.probe = cp.array(self.experimentalData.probe, cp.complex64)
        #self.optimizable.Imeasured = cp.array(self.optimizable.Imeasured)

        # ePIE parameters
        self.logger.info('Detector error shape: %s', self.detectorError.shape)
        self.detectorError = cp.array(self.detectorError)

        # proapgators to GPU
        if self.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
        elif self.propagator == 'ASP' or self.propagator == 'polychromeASP':
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)
        elif self.propagator =='scaledASP' or self.propagator == 'scaledPolychromeASP':
            self.optimizable.Q1 = cp.array(self.optimizable.Q1)
            self.optimizable.Q2 = cp.array(self.optimizable.Q2)

        # other parameters
        if self.backgroundModeSwitch:
            self.background = cp.array(self.background)
        if self.absorbingProbeBoundary:
            self.probeWindow = cp.array(self.probeWindow)

