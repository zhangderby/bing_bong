from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.propagation import (
    focus_fixed_sampling,
    angular_spectrum
)
from prysm.propagation import Wavefront as WF
from prysm.geometry import circle,spider
from prysm.polynomials import (
    noll_to_nm,
    zernike_nm,
    zernike_nm_seq,
    hopkins,
    sum_of_2d_modes
)

from prysm.fttools import (fourier_resample,
                           crop_center,
                           pad2d,
)

from prysm import mathops, conf
mathops.set_backend_to_cupy()
# conf.config.precision = 32
from astropy.io import fits

from prysm.mathops import (np,
                           fft,
                           interpolate,
                           ndimage)

import matplotlib.pyplot as plt

import sys
from lina.phase_retrieval import ADPhaseRetireval, ParallelADPhaseRetrieval
from stppsf import wcc_batoid

from psd_utils import PSDUtils

from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator


global WFE_MAPS_PATH
WFE_MAPS_PATH = '/home/derbyk/src/stp_wfe_maps/telescope/'

global M1_MAP_FN
M1_MAP_FN = 'M1_goal_map.fits'

global M2_MAP_FN
M2_MAP_FN = 'M2_map.fits'

global M3_MAP_FN
M3_MAP_FN = 'M3_map.fits'

global M4_MAP_FN
M4_MAP_FN = 'M4_map.fits'


class WCC:

    def __init__(self, 
                 npix_beam,
                 npix_focal,
                 wvls, 
                 wvl_weights, 
                 src_magnitudes, 
                 src_positions,):

        ##### MODEL PARAMETERS #####

        # user-defined
        self.npix_beam = npix_beam                # pix
        self.npix_focal = npix_focal                # pix
        self.wvls = wvls                            # um
        self.wvl_weights = wvl_weights              # must sum to 1
        self.src_magnitudes = src_magnitudes        # Rmag
        self.src_positions = src_positions          # arcsec

        # stuff that will be imported from toml/yaml
        self.D_obs = 1300                           # M2 obsurcation diameter [mm]
        self.D_spider = 150                         # spider diameter [mm]

        self.f_m1 = 
        self.f_m2 = 
        self.f_m3 = 
        self.f_m4 = np.inf                          # M4 focal length [mm]
        
        self.BD_m1 = 6500                           # beam diameter on M1 [mm]
        self.dx_m1 = self.BD_m1 / npix_beam        # beam pixelscale on M1 [mm/pix]
        self.BD_m2 = 1.0882381890e02 * 2            # beam diameter on M2 [mm]
        self.dx_m2 = self.BD_m2 / npix_beam        # beam pixelscale on M2 [mm/pix]
        self.BD_m3 = 2.8372636491e01 * 2            # beam diameter on M3 [mm]
        self.dx_m3 = self.BD_m3 / npix_beam        # beam pixelscale on M3 [mm/pix]
        self.BD_m4 = 
        self.dx_m4 = self.BD_m4 / npix_beam        # beam pixelscale on M4 [mm/pix]

        ##### CONSTRUCT MODEL #####

        # M1
        self.x_m1, self.y_m1 = make_xy_grid(npix_beam, self.dx_m1, self.BD_m1)
        self.r_m1, self.t_m1 = cart_to_polar(self.x_m1, self.y_m1)

        # aperture
        s1 = spider(1, self.D_spider, self.x_m1, self.y_m1, rotation=0)
        s2 = spider(1, self.D_spider, self.x_m1, self.y_m1, rotation=120)
        s3 = spider(1, self.D_spider, self.x_m1, self.y_m1, rotation=240)
        spiders = s1 & s2 & s3
        self.aperture = (circle(self.BD_m1 / 2, self.r_m1) ^ circle(self.D_obs / 2, self.r_m1)) & spiders

        # M2
        self.x_m2, self.y_m2 = make_xy_grid(npix_beam, self.dx_m2, self.BD_m2)
        self.r_m2, self.t_m2 = cart_to_polar(self.x_m2, self.y_m2)

        # M3
        self.x_m3, self.y_m3 = make_xy_grid(npix_beam, self.dx_m3, self.BD_m3)
        self.r_m3, self.t_m3 = cart_to_polar(self.x_m3, self.y_m3)

        # M4
        self.x_m4, self.y_m4 = make_xy_grid(npix_beam, self.dx_m4, self.BD_m4)
        self.r_m4, self.t_m4 = cart_to_polar(self.x_m4, self.y_m4)

    def _wcc_factory(cls):

        data = fits.getdata(WFE_MAPS_PATH + 'M1_goal_map.fits')

        xi = yi = np.linspace(-6.42 * 1.05 / 2, 6.52 * 1.05 / 2, data.shape[0]).get()
        interp = RegularGridInterpolator((xi, yi), data)

        xf = yf = np.linspace(-6.42 / 2, 6.42 / 2, 512)
        xf, yf = np.meshgrid(xf, yf, indexing='ij')
        interp_data = interp((xf.get(), yf.get()))

        return cls()





    def create_wavefront(wvl, src_magnitude, src_position):

        wavefront = WF.from_amp_and_phase(self.aperture, None, wvl, self.dx_pupil)

        return wavefront
    
    def get_field_aberration(src_position):

        return None


    def _fwd(self, system, wvl, src_magnitude, src_position):

        wf = self.create_wavefront(wvl, src_magnitude, src_position)

        for optic in system.optics:

            complex_screen = WF.from_amp_and_phase(optic.get_amplitude(wvl), optic.get_phase(wvl), wvl)

            wf *= complex_screen

            wf = angular_spectrum(wf, wvl, optic.dx)

        detector_intensity = np.abs(wf) ** 2

        return detector_intensity


    def snap(self, t):

        images = []
            
        for src_magnitude, src_position in zip(self.src_magnitudes, self.src_positions):

            detector_intensities = []

            for wvl in self.wvls:
                
                detector_intensity = self._fwd(self, wvl, src_magnitude, src_position)

                detector_intensities.append(detector_intensity)

            psf = sum_of_2d_modes(detector_intensities, self.wvl_weights)

            jittered_psf = self._pointing_factory(psf)

            image = self.integrate_and_read_out(jittered_psf, t)

            images.append(image)

        return images