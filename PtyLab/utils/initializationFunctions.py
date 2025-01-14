import logging
import numpy as np
from PtyLab.utils.utils import circ, fft2c, ifft2c
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale


def initialProbeOrObject(shape, type_of_init, data, logger: logging.Logger = None):
    """
    Initialization objects are created for the reconstruction. Currently
    implemented:
        ones - every element is set to 1 + random noise
        circ - same as 'ones' but with a circular boundary constraint
        upsampled - upsampled low-resolution estimate (used for FPM)

    Random noise is added to the arrays to enforce linear independence required
    for orthogonalization of modes

    :return:
    """
    if type(type_of_init) is np.ndarray:  # it has already been run
        if logger is not None:
            logger.warning(
                "initialObjectOrProbe was called but the object has already "
                "been initialized. Skipping."
            )
        return type_of_init

    if type_of_init not in ["circ", "rand", "gaussian", "ones", "upsampled"]:
        raise NotImplementedError()

    if type_of_init == "ones":
        return np.ones(shape) + 0.001 * np.random.rand(*shape)

    if type_of_init == "circ":
        try:
            # BUG: This only works for the probe, not for the object
            pupil = circ(data.Xp, data.Yp, data.data.entrancePupilDiameter)

            # pupil = circ(data.Xp, data.Yp, data.Xp.max()/2)
            # soften the edges a bit
            from scipy import ndimage

            pupil = ndimage.gaussian_filter(
                pupil.astype(np.float64), 0.05 * data.Xp.shape[-1]
            )
            return np.ones(shape, dtype=np.complex64) * pupil + 0.001 * np.random.rand(
                *shape
            )

        except AttributeError as e:
            raise AttributeError(
                e, "probe/aperture/entrancePupilDiameter was not defined"
            )

    if type_of_init == "upsampled":
        low_res = ifft2c(np.sqrt(np.mean(data.data.ptychogram, 0)))
        pad_size = (int((data.No - data.Np) / 2), int((data.No - data.Np) / 2))
        upsampled = np.pad(
            low_res, pad_size, mode="constant", constant_values=0
        )  # * data.No / data.Np
        return np.ones(shape) * upsampled
