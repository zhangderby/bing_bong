import numpy as np
import batoid as btd
import time
from copy import deepcopy
import cupy as cp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from scipy.interpolate import griddata
from datetime import date
import time
from itertools import chain, compress
import matplotlib.colors as mcolors
import poppy as pp

class model():

    def __init__(self,
                 wavelength=650e-9,
                 npix_pupil=512):
        
        self.wavelength = wavelength
        self.npix_pupil = npix_pupil

        # field bias
        self.field_bias = 0.17

        # M1 prescription
        self.M1_RoC = -16.256
        self.M1_conic = -0.995357
        self.M1_innerD = 1.38
        self.M1_outerD = 6.42

        # M1 position
        self.M1_dx = 0
        self.M1_dy = 0
        self.M1_dz = 0
        self.M1_tx = 0
        self.M1_ty = 0

        # M2 prescription
        self.M2_RoC = -1.660575
        self.M2_conic = -1.566503
        self.M2_outerD = 0.7147252

        # M2 position
        self.M2_dx = 0
        self.M2_dy = 0
        self.M2_dz = -7.400
        self.M2_tx = 0
        self.M2_ty = 0

        # M3 prescription
        self.M3_RoC = -1.830517
        self.M3_conic = -0.7180517
        self.M3_outerD = 0.920733

        # M3 position
        self.M3_dx = 0
        self.M3_dy = 0
        self.M3_dz = 0.05
        self.M3_tx = 0
        self.M3_ty = 0

        # M4 prescription
        self.M4_outerD = 0.0991468

        # M4 position
        self.M4_dx = 0
        self.M4_dy = 0
        self.M4_dz = -0.983629
        self.M4_tx = 0
        self.M4_ty = 0

        # detector prescription
        self.det_outerD = 0.9

        # detector position
        self.det_dx = 0
        self.det_dy = 0
        self.det_dz = 0.237581
        self.det_tx = 0
        self.det_ty = 0

        self.init_model()

    def init_model(self):

        # build batoid model
        M1coord = defineCoordinate(posX=self.M1_dx, posY=self.M1_dy, posZ=self.M1_dz, anglX=self.M1_tx, anglY=self.M1_ty).local()
        # M1 = optic('M1',type='Quadric', RoC=self.M1_RoC, conic=self.M1_conic, inDiam=self.M1_innerD, outDiam=self.M1_outerD).mirrorWithSpider(width=0.15, height=4)
        M1 = optic('M1',type='Quadric', RoC=self.M1_RoC, conic=self.M1_conic, outDiam=self.M1_outerD, coordSys=M1coord).mirror()

        M2coord = defineCoordinate(posX=self.M2_dx, posY=self.M2_dy, posZ=self.M2_dz, anglX=self.M2_tx, anglY=self.M2_ty).local()
        M2 = optic('M2', type='Quadric', RoC=self.M2_RoC, conic=self.M2_conic, outDiam=self.M2_outerD, coordSys=M2coord).mirror()

        M3coord = defineCoordinate(posX=self.M3_dx, posY=self.M3_dy, posZ=self.M3_dz, anglX=self.M3_tx, anglY=self.M3_ty).local()
        M3 = optic('M3', type='Quadric', RoC=self.M3_RoC, conic=self.M3_conic, outDiam=self.M3_outerD, coordSys=M3coord).mirror()

        M4coord = defineCoordinate(posX=self.M4_dx, posY=self.M4_dy, posZ=self.M4_dz, anglX=self.M4_tx, anglY=self.M4_ty).local()
        M4 = optic('M4', outDiam=self.M4_outerD, coordSys=M4coord).flatMirror()

        Dcoord = defineCoordinate(posX=self.det_dx, posY=self.det_dy, posZ=self.det_dz, anglX=self.det_tx, anglY=self.det_ty).local()
        D = optic('D', outDiam=self.det_outerD, coordSys=Dcoord).detector()

        self.osys = build.compoundOptic(M1, M2, M3, M4, D, pupilSize=6.46, backDist=7.5, EPcoord=defineCoordinate(posZ=0).local())
        
    def add_motion(self, 
                   M1_dx=0, M1_dy=0, M1_dz=0, M1_tx=0, M1_ty=0,
                   M2_dx=0, M2_dy=0, M2_dz=0, M2_tx=0, M2_ty=0,
                   M3_dx=0, M3_dy=0, M3_dz=0, M3_tx=0, M3_ty=0,
                   M4_dx=0, M4_dy=0, M4_dz=0, M4_tx=0, M4_ty=0,
                   det_dx=0, det_dy=0, det_dz=0, det_tx=0, det_ty=0,):

        # M1 position
        self.M1_dx += M1_dx
        self.M1_dy += M1_dy
        self.M1_dz += M1_dz
        self.M1_tx += M1_tx
        self.M1_ty += M1_ty

        # M2 position
        self.M2_dx += M2_dx
        self.M2_dy += M2_dy
        self.M2_dz += M2_dz
        self.M2_tx += M2_tx
        self.M2_ty += M2_ty

        # M3 position
        self.M3_dx += M3_dx
        self.M3_dy += M3_dy
        self.M3_dz += M3_dz
        self.M3_tx += M3_tx
        self.M3_ty += M3_ty

        # M4 position
        self.M4_dx += M4_dx
        self.M4_dy += M4_dy
        self.M4_dz += M4_dz
        self.M4_tx += M4_tx
        self.M4_ty += M4_ty

        # detector position
        self.det_dx += det_dx
        self.det_dy += det_dy
        self.det_dz += det_dz
        self.det_tx += det_tx
        self.det_ty += det_ty

        self.init_model()

    def reset_motion(self):

        # M1 position
        self.M1_dx = 0
        self.M1_dy = 0
        self.M1_dz = 0
        self.M1_tx = 0
        self.M1_ty = 0

        # M2 position
        self.M2_dx = 0
        self.M2_dy = 0
        self.M2_dz = -7.400
        self.M2_tx = 0
        self.M2_ty = 0

        # M3 position
        self.M3_dx = 0
        self.M3_dy = 0
        self.M3_dz = 0.05
        self.M3_tx = 0
        self.M3_ty = 0

        # M4 position
        self.M4_dx = 0
        self.M4_dy = 0
        self.M4_dz = -0.983629
        self.M4_tx = 0
        self.M4_ty = 0

        # detector position
        self.det_dx = 0
        self.det_dy = 0
        self.det_dz = 0.237581
        self.det_tx = 0
        self.det_ty = 0

        self.init_model()
        
    def get_opd(self, fieldX=0, fieldY=0, PTTremove=True, plot=False):
        
        # apply field bias
        fieldY += self.field_bias

        # get opd data
        infoTelescope = details(self.osys, fieldX=fieldX, fieldY=fieldY, wavelength=self.wavelength)
        self.wf = infoTelescope.wavefront(npx=self.npix_pupil)
        
        # plot if desired
        if plot:
            plot.wavefront(self.wf, self.wavelength, [fieldX, fieldY-self.field_bias])
            # print(f'Tilt Removed:  X = {np.round(removed[0], decimals=4)}, Y = {np.round(removed[1], decimals=4)} waves\n')

        # what the fuck is this piece of shit masked array object
        what =  self.wf
        the = what.data * ~what.mask
        fuck = the * self.wavelength

        # get a mask of the pupil
        mask = fuck != 0

        # the batoid pupil isn't quite a unit circle so for now I'm zernike decomposing with poppy
        # and recomposing that onto a 2D unit circle
        zerns = pp.zernike.decompose_opd_nonorthonormal_basis(cp.asarray(fuck), aperture=cp.asarray(mask), nterms=50)

        if PTTremove:
            zerns[:3] = 0

        opd = pp.zernike.compose_opd_from_basis(zerns, outside=0).get()

        return opd
    
    def get_zernikes(self, fieldX=0, fieldY=0, nterms=37):
        
        # apply field bias
        fieldY += self.field_bias

        # get opd data
        infoTelescope = details(self.osys, fieldX=fieldX, fieldY=fieldY, wavelength=self.wavelength)
        self.wf = infoTelescope.wavefront(npx=self.npix_pupil)

        # what the fuck is this piece of shit masked array object
        what =  self.wf
        the = what.data * ~what.mask
        fuck = the * self.wavelength

        # get a mask of the pupil
        mask = fuck != 0

        # the batoid pupil isn't quite a unit circle so for now I'm zernike decomposing with poppy
        # and recomposing that onto a 2D unit circle
        zerns = pp.zernike.decompose_opd_nonorthonormal_basis(cp.asarray(fuck), aperture=cp.asarray(mask), nterms=nterms)

        return zerns
        
class rotation:
    """Abstract base class defining 3D rotation matrices.
    """
    def __init__(self, anglX = 0., anglY = 0., anglZ = 0.):
        """
        Parameters
        ----------
        anglX : float
            Angle around X-axis (in degrees). Default is 0.
        anglY : float
            Angle around Y-axis (in degrees). Default is 0.
        anglZ : float
            Angle around Z-axis (in degrees). Default is 0.
        """
        self.anglX = anglX
        self.anglY = anglY
        self.anglZ = anglZ
    
    def Rx(self):
        """The function defining the 3D rotation matrix around X-axis; Rx(anglX).

        Parameters
        ----------
        anglX : float
            Angle of rotation in degree.

        Returns
        -------
        Rx : array_like, shape (3,3)
             Rotation matrix around X-axis.
        """
        theta = np.deg2rad(self.anglX)
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    def Ry(self):
        """The function defining the 3D rotation matrix around Y-axis; Ry(anglY).

        Parameters
        ----------
        anglY : float
            Angle of rotation in degree.

        Returns
        -------
        Ry : array_like, shape (3,3)
             Rotation matrix around Y-axis.
        """
        theta = np.deg2rad(self.anglY)
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
    
    def Rz(self):
        """The function defining the 3D rotation matrix around Z-axis; Rz(anglZ).

        Parameters
        ----------
        anglZ : float
            Angle of rotation in degree.

        Returns
        -------
        Rz : array_like, shape (3,3)
             Rotation matrix around Z-axis.
        """
        theta: float = np.deg2rad(self.anglZ)
        return np.array([[np.cos(theta), -np.sin(theta),0 ],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    
class defineCoordinate:
    """Abstract base class defining coordinates for individual optical elements.
    """
    def __init__(self, posX=0, posY=0, posZ=0, anglX=0, anglY=0, anglZ=0):
        """
        Parameters
        ----------
        posX : float
            X position of the origin (in meters). Default is 0.
        posY : float
            Y position of the origin (in meters). Default is 0.
        posZ : float
            Z position of the origin (in meters). Default is 0.
        anglX : float
            Angle around X-axis (in degrees). Default is 0.
        anglY : float
            Angle around Y-axis (in degrees). Default is 0.
        anglZ : float
            Angle around Z-axis (in degrees). Default is 0.
        """
        self.posX = posX
        self.posY = posY
        self.posZ = posZ
        self.anglX = anglX
        self.anglY = anglY
        self.anglZ = anglZ
    
    def local(self, order='xyz'):
        """The function defining local coordinates for an individual optical element with a specific rotation order.

        Parameters
        ----------
        order : string, len (3)
            Order of the rotations to be applied.
            ``xyz`` (default), assumes we multiply rotation matrices in the following order: Rx * Ry * Rz

        Returns
        -------
        coordSys : batoid.CoordSys
             Coordinate system centered in (x,y,z) and rotated by (anglX,anglY,anglZ).
        """
        x, y, z = self.posX, self.posY, self.posZ
        R = rotation(self.anglX, self.anglY, self.anglZ)
        Rx, Ry, Rz = R.Rx(), R.Ry(), R.Rz()
        if order.casefold() == 'xyz':
            return btd.CoordSys(origin=np.array([x, y, z]), rot=np.matmul(Rx, Ry, Rz))
        elif order.casefold() == 'yzx':
            return btd.CoordSys(origin=np.array([x, y, z]), rot=np.matmul(Ry, Rz, Rx))
        elif order.casefold() == 'zyx':
            return btd.CoordSys(origin=np.array([x, y, z]), rot=np.matmul(Rz, Ry, Rx))
        
class utils:
    """Abstract base class redefining some specific native Batoid's function to match ZEMAX analysis.
    """   
    def __init__(self): 
        pass
        
    def zemaxToDirCos(u, v):
        """Convert Zemax field angles u,v to direction cosines.
    
        Parameters
        ----------
        u, v : float
            Zemax field angles in radians.
    
        Returns
        -------
        alpha, beta, gamma : float
            Direction cosines (unit vector projected onto x, y, z in order)
    
        Notes
        -----
        The tangent plane reference is at (u,v) = (0,0), which corresponds to
        (alpha, beta, gamma) = (0, 0, 1) (a ray coming directly from above).  The
        orientation is such that vx (vy) is positive when u (v) is negative.
    
        The Zemax field angle convention is not rotationally invariant.  The
        z-direction cosine for (u, v) = (0, 1) does not equal the z-direction
        cosine for (u, v) = (0.6, 0.8).
        """
        tanu = np.tan(u)
        tanv = np.tan(v)
        norm = np.sqrt(1 + tanu*tanu + tanv*tanv)
        return tanu/norm, tanv/norm, 1/norm

    def dirCosToZemax(alpha, beta, gamma):
        """Convert direction cosines to Postel azimuthal equidistant tangent plane
        projection.

        Parameters
        ----------
        alpha, beta, gamma : float
            Direction cosines (unit vector projected onto x, y, z in order)

        Returns
        -------
        u, v : float
            Postel tangent plane coordinates in radians.

        Notes
        -----
        The tangent plane reference is at (u,v) = (0,0), which corresponds to
        (alpha, beta, gamma) = (0, 0, +1) (a ray coming directly from above).  The
        orientation is such that vx (vy) is positive when u (v) is positive.

        The Zemax field angle convention is not rotationally invariant.  The
        z-direction cosine for (u, v) = (0, 1) does not equal the z-direction
        cosine for (u, v) = (0.6, 0.8).
        """
        return np.arctan(-alpha/gamma), np.arctan(-beta/gamma)

    def _closestApproach(P, u, Q, v):
        """Compute position along line P + u t of closest approach to line Q + v t

        Parameters
        ----------
        P, Q : ndarray
            Points on lines
        u, v : ndarray
            Direction cosines of lines

        Returns
        -------
        Pc : ndarray
            Closest approach point in meters.
        """
        # Follows http://geomalgorithms.com/a07-_distance.html
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        w0 = P - Q
        d = np.dot(u, w0)
        e = np.dot(v, w0)
        den = a*c - b*b
        if den == 0:
            raise ValueError("Lines are parallel")
        sc = (b*e - c*d)/den
        return P + sc*u

    def exitPupilPos(optic, wavelength, smallAngle=np.deg2rad(1./3600), **kwargs):
        """Compute position of the exit pupil.

        Traces a collection of small angle chief rays into object space, and then
        finds the mean of their closest approaches to the optic axis.  Possibly only
        accurate for axially symmetric optical systems.

        Parameters
        ----------
        optic : batoid.Optic
            Optical system
        wavelength : float
            Wavelength in meters
        smallAngle : float, optional
            Angle in radians from which to search for the exit pupil position.

        Returns
        -------
        Location of exit pupil in global coordinates (in meters).
        """
        thx = np.array([0, 0, smallAngle, -smallAngle])
        thy = np.array([smallAngle, -smallAngle, 0, 0])
        rays = btd.RayVector.fromFieldAngles(thx, thy, optic=optic, wavelength=wavelength, **kwargs)
        optic.trace(rays)
        rays.toCoordSys(btd.globalCoordSys)
        # Find the optical axis
        OA = btd.RayVector.fromStop(x=0, y=0,
                                    optic=optic,
                                    wavelength=wavelength,
                                    dirCos=utils.zemaxToDirCos(np.deg2rad(0.0), np.deg2rad(0.0)),
                                    flux=1.)
        optic.trace(OA)
        OA.toCoordSys(btd.globalCoordSys)
        # Assume last intersection places rays into "object" space.
        # Now find closest approach to optic axis.
        ps = []
        for ray in rays:
            ps.append(utils._closestApproach(ray.r[0], ray.v[0],
                                             OA.r[0], OA.v[0]
                                            )
                     )
        return np.mean(ps, axis=0)

    def exitPupilDir(optic, wavelength):
        # Find the optical axis
        OA = btd.RayVector.fromStop(x=0, y=0,
                                    optic=optic,
                                    wavelength=wavelength,
                                    dirCos=utils.zemaxToDirCos(np.deg2rad(0.0), np.deg2rad(0.0)),
                                    flux=1.)
        optic.trace(OA)
        OA.toCoordSys(btd.globalCoordSys)
        return OA.v[0]
        

    def XPdiam(opt, theta_x=0, theta_y=0, wavelength=633e-9):
        nx = 256
        dirCos = utils.zemaxToDirCos(np.deg2rad(theta_x), np.deg2rad(theta_y))
        rays = btd.RayVector.asGrid(optic=opt, wavelength=wavelength, nx=nx, dirCos=dirCos)
        
        XPloc = utils.exitPupilPos(opt, wavelength=wavelength)
        XPcoord = defineCoordinate(posX=XPloc[0], posY=XPloc[1], posZ=XPloc[-1]).local()
        originalD = opt.items[-1]
        XP = optic('XP', inDiam=originalD.inDiam, outDiam=originalD.outDiam, coordSys=XPcoord).detector()
        newSysAtXP = build.compoundOptic(*opt.items[0:-1], XP,
                                         pupilSize=opt.pupilSize, backDist=opt.backDist,
                                         EPcoord=opt.stopSurface.coordSys)
        newSysAtXP.trace(rays, reverse=False)
        w = np.where(1-rays.vignetted)[0]

        x, y = rays.x[w]-np.mean(rays.x[w]), rays.y[w]-np.mean(rays.y[w])
        r = np.sqrt(x**2 + y**2)

        X = np.abs(x)
        indX = np.argwhere(X <= X[X.argmin()])
        b = np.max(y[indX])-np.min(y[indX])
        #a = np.max(x[indY])-np.min(x[indY])
        #R = 2 * np.max(r)

        #plt.figure()
        #plt.plot(x, y, '.')
        #plt.grid()
        #plt.axis('equal')
        #plt.show()
        
        return b

    def zernike(optic, theta_x, theta_y, wavelength,
                projection='postel', nx=32,
                sphereRadius=None, reference='chief', jmax=22, eps=0.0):
        import galsim

        #dirCos = btd.utils.fieldToDirCos(theta_x, theta_y, projection=projection)
        dirCos = utils.zemaxToDirCos(theta_x, theta_y)
        rays = btd.RayVector.asGrid(optic=optic, wavelength=wavelength,
                                    nx=nx, dirCos=dirCos)
        
        # Propagate to entrance pupil to get positions
        epRays = rays.toCoordSys(optic.stopSurface.coordSys)
        optic.stopSurface.surface.intersect(epRays)
        orig_x = np.array(epRays.x).reshape(nx, nx)
        orig_y = np.array(epRays.y).reshape(nx, nx)

        wf = utils.wavefront(optic, theta_x, theta_y, wavelength, nx=nx,
                             projection=projection, sphereRadius=sphereRadius,
                             reference=reference)
        wfarr = wf.array
        w = ~wfarr.mask

        basis = galsim.zernike.zernikeBasis(jmax, orig_x[w], orig_y[w],
                                            R_outer=optic.pupilSize/2, R_inner=optic.pupilSize/2*eps
                                           )
        coefs, _, _, _ = np.linalg.lstsq(basis.T, wfarr[w], rcond=-1)
        # coefs[0] is meaningless, so always set to 0.0 for comparison consistency
        coefs[0] = 0.0
        return np.array(coefs)

    def zernikeFromWavefront(WF, jmax=22):
        import galsim

        wfarr = WF.data
        w = ~WF.mask

        N = np.shape(wfarr)[0]

        val = np.linspace(-1, 1, N)
        x, y = np.meshgrid(val, val)
        
        basis = galsim.zernike.zernikeBasis(jmax, x[w], y[w],
                                            R_outer=1, R_inner=0
                                           )
        coefs, _, _, _ = np.linalg.lstsq(basis.T, wfarr[w], rcond=-1)
        # coefs[0] is meaningless, so always set to 0.0 for comparison consistency
        coefs[0] = 0.0
        return np.array(coefs)

    ########

    def wavefront(optic, theta_x, theta_y, wavelength,
                  projection='postel', nx=32, sphereRadius=None, reference='chief'):
        
        dirCos = utils.zemaxToDirCos(theta_x, theta_y)
        rays = btd.RayVector.asGrid(optic=optic, wavelength=wavelength,
                                    nx=nx, dirCos=dirCos)
        
        if sphereRadius is None:
            sphereRadius = optic.sphereRadius

        optic.trace(rays)
        
        cridx, point = utils.referencePoint(rays, nx=nx)
            
        # Place vertex of reference sphere one radius length away from the
        # intersection point.  So transform our rays into that coordinate system.
        targetCoordSys = rays.coordSys.shiftLocal(point+np.array([0, 0, sphereRadius])
                                                 )
        rays.toCoordSys(targetCoordSys)

        sphere = btd.Sphere(-sphereRadius)
        sphere.intersect(rays)

        t0 = rays.t[cridx]
        OPD = (t0-rays.t)/wavelength
            
        arr = np.ma.masked_array(OPD,
                                 mask=rays.vignetted).reshape(nx, nx)
        if (nx%2) == 0:
            primitiveU = np.vstack([[optic.pupilSize/(nx-2), 0],
                                    [0, optic.pupilSize/(nx-2)]])
        else:
            primitiveU = np.vstack([[optic.pupilSize/(nx-1), 0],
                                    [0, optic.pupilSize/(nx-1)]])
        return btd.Lattice(arr, primitiveU)
        
    ########
    ########
    def referencePoint(rays, dx=0, dy=0, dz=0, 
                       nx=32):
        
        cridx = (nx//2)*nx+nx//2 if (nx%2)==0 else (nx*nx-1)//2
        point = rays.r[cridx]
        
        return cridx, point + np.array([dx, dy, dz])


    def wavefrontRMS(params, optic, theta_x, theta_y, wavelength, nx, sphereRadius):

        dx, dy, dz = params[0], params[1], params[2]
        dx=0
        dy=0
        
        dirCos = utils.zemaxToDirCos(theta_x, theta_y)
        rr = btd.RayVector.asGrid(optic=optic, wavelength=wavelength,
                                  nx=nx, dirCos=dirCos)

        if sphereRadius is None:
            sphereRadius = optic.sphereRadius

        optic.trace(rr)

        cridx, point = utils.referencePoint(rr, dx=dx, dy=dy, dz=dz, nx=nx)
            
        # Place vertex of reference sphere one radius length away from the
        # intersection point.  So transform our rays into that coordinate system.
        targetCoordSys = rr.coordSys.shiftLocal(point+np.array([0, 0, sphereRadius])
                                                 )
        rr.toCoordSys(targetCoordSys)

        sphere = btd.Sphere(-sphereRadius)
        sphere.intersect(rr)

        t0 = rr.t[cridx]
        arr = np.ma.masked_array(data=(t0-rr.t)/wavelength,
                                 mask=rr.vignetted).reshape(nx, nx)
        print('merit :', np.std(arr), end="\r")
        return np.std(arr)

    def optiTEST(optic, theta_x, theta_y, wavelength, nx, sphereRadius):

        #global optic, theta_x, theta_y, wavelength, nx, sphereRadius
        initial_guess = np.zeros(3)
        result = scipy.optimize.minimize(utils.wavefrontRMS, x0=initial_guess, args=(optic, theta_x, theta_y, wavelength, nx, sphereRadius),
                                        bounds=((-10e-6, 10e-6),
                                                (-10e-6, 10e-6),
                                                (-10e-6, 10e-6),), tol=1e-7)

        if result.success:
            best_params = result.x
            return best_params
        else:
            raise ValueError(result.message)
    
class optic:
    """Abstract base class defining some optical elements (only detectors and mirrors for now).
    """
    def __init__(self, name, type='Plane', 
                 RoC=-1, conic=0, inDiam=0, outDiam=1, indx=1.0,
                 coordSys=defineCoordinate().local()
                ):
        """
        Parameters
        ----------
        name : string
            Name of the optical surface.
        type : string
            Defining the shape of the surface. Default is ``Plane``.
        RoC : float
            Radius of curvature of the surface (in meters). Default is ``-1``.
        conic : float
            Conic constant of the surface. Default is ``0``.
        inDiam : float
            Inner diameter of the surface (in meters). Default is ``0``.
        outDiam : float
            Outter diameter of the surface (in meters). Default is ``1``.
        indx : float
            Refractive index before and after the surface. Default is ``1.0``.
        coordSys : batoid.CoordSys
            Coordinate system associated with the surface. Default is ``(0,0,0);(1,0,0),(0,1,0),(0,0,1)``.
        """
        self.name = name
        self.type = type
        self.RoC = RoC
        self.conic = conic
        self.inDiam = inDiam
        self.outDiam = outDiam
        self.indx = indx
        self.coordSys = coordSys

    def obscMirror(self):
        """The function defining the oscuration of a mirror.

        Parameters
        ----------
        
        Returns
        -------
        coordSys : batoid.Obscuration
             Annulus light going through depending on inDiam and outDiam. 
             If inDiam=0, then the light goes through a circle of a diameter of outDiam.
             ``+10`` represent the upper bound of obscuration's circle. It is arbitrary but it avoids straylight.
        """
        inDiam, outDiam = self.inDiam, self.outDiam
        if inDiam == 0:
            return btd.ObscAnnulus(outDiam/2, outDiam/2+10)
        else:
            return btd.ObscUnion(btd.ObscCircle(inDiam/2), btd.ObscAnnulus(outDiam/2, outDiam/2+10))

    def obscMirrorWithSpider(self, width, height):
        """The function defining the oscuration of a mirror with a 3-arms spider.

        Parameters
        ----------
        width : float
            Arms' width (in meters).
        height : float
            Arms' height (in meters).
        
        Returns
        -------
        coordSys : batoid.Obscuration
             Annulus light going through depending on inDiam and outDiam.
             3 arms obscur this annulus.
        """
        obsc = btd.ObscUnion(self.obscMirror(), 
                             btd.ObscRectangle(width=width, height=height, 
                                               x=height/2, y=0.0, theta=np.pi/2),
                             btd.ObscRectangle(width=width, height=height, 
                                               x=-height/2*np.sin(np.pi/6), y=-4/2*np.cos(np.pi/6), theta=-np.pi/6),
                             btd.ObscRectangle(width=width, height=height, 
                                               x=-height/2*np.sin(5*np.pi/6), y=-4/2*np.cos(5*np.pi/6), theta=-5*np.pi/6))
        return obsc
            
    def flatMirror(self):
        """The function defining a flat mirror.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Flat mirror.
        """
        type = self.type
        if type.casefold() == 'plane':
            obj = btd.Mirror(name=self.name,
                             surface=btd.Plane(), 
                             obscuration=self.obscMirror(),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=btd.ConstMedium(self.indx),
                             outMedium=btd.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj    

    def mirror(self):
        """The function defining a simple mirror with an annulus bundle of rays.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Mirror following a ``Quadric`` desciption, meaning RoC + conic.
        """
        type = self.type 
        if type.casefold() == 'quadric':
            obj = btd.Mirror(name=self.name,
                             surface=btd.Quadric(self.RoC, self.conic), 
                             obscuration=self.obscMirror(),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=btd.ConstMedium(self.indx),
                             outMedium=btd.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj 

    def mirrorWithSpider(self, width=0, height=0):
        """The function defining a simple mirror with an spider.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Mirror following a ``Quadric`` desciption, meaning RoC + conic.
        """
        type = self.type 
        if type.casefold() == 'quadric':
            obj = btd.Mirror(name=self.name,
                             surface=btd.Quadric(self.RoC, self.conic), 
                             obscuration=self.obscMirrorWithSpider(width, height),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=btd.ConstMedium(self.indx),
                             outMedium=btd.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj 

    def detector(self):
        """The function defining a detector. The detector should be the last surface in optical systems.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Detector. It should be the last surface in optical systems.
        """
        type = self.type
        if type.casefold() == 'plane':
            obj = btd.Detector(btd.Plane(), 
                               coordSys=self.coordSys, 
                               name=self.name,
                               inDiam=self.inDiam,
                               outDiam=self.outDiam)
            return obj

class build:
    """Abstract base class defining the building of an optical system.
       Note that, for now, it is only possible to define the stop as an entrance pupil.
    """
    def stopSurface(coordSys=defineCoordinate().local()):
        """The function defining the stop surface as being the entrance pupil.

        Parameters
        ----------
        coordSys : batoid.CoordSys
            Location and orientation of the stop. Defalt is ``(0,0,0);(1,0,0),(0,1,0),(0,0,1)``.
        
        Returns
        -------
        sys : batoid.Optic
             Interface of this stop.
        """
        return btd.Interface(btd.Plane(), coordSys=coordSys)
    
    def compoundOptic(*args, pupilSize, backDist=40, EPcoord):
        """The function defining a compound of optical surfaces as an optical system.

        Parameters
        ----------
        pupilSize : float
            Pupil's diameter (in meters).
        backDist : float
            Thickness of the first surface (in meters). Default is ``40``.
            Describes how far rays are coming to the first optical surface.
        
        Returns
        -------
        sys : batoid.Optic
             Compound of optical surfaces forming an optical system.
        """
        optics = [item for item in args]
        sys = btd.CompoundOptic(optics,
                                backDist=backDist, 
                                stopSurface=build.stopSurface(coordSys=EPcoord),
                                pupilSize = pupilSize)
        return sys
    
class ray:
    """Abstract base class defining different ray possibilities and computations.
    """
    def __init__(self,
                 opticalSys, fieldX=0, fieldY=0, wavelength=633e-9, nbOfRays=1
                ):
        """
        Parameters
        ----------
        opticalSys : batoid.Optic
            Optical system as a single optic or a compound optic.
        fieldX : float
            Field over X (in degrees). Default is ``0``.
        fieldY : float
            Field over Y (in degrees). Default is ``0``.
        wavelength : float
            Wavelength for this ray (in meters). Default is ``633e-9``.
        nbOfRays : integer
            Number of rays to create. Default is ``1``.
        """
        self.opticalSys = opticalSys
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.wavelength = wavelength
        self.nbOfRays = nbOfRays

    def createPolarBundleOfRays(self):
        """The function defining x and y coordinates of each ray.
           This is a uniform distribution of rays over angles and radii of the pupil.

        Parameters
        ----------
        
        Returns
        -------
        x : array_like
             X coordinates.
        y : array_like
             Y coordinates.
        """
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        th = 2 * np.pi * np.random.rand(int(N))
        cth, sth = np.cos(th), np.sin(th)
        d = (size) * np.random.rand(int(N))
        x, y = (d/2) * cth, (d/2) * sth
        return x, y

    def createGridBundleOfRays(self):
        """The function defining x and y coordinates of each ray.
           This is a uniform distribution of rays over the tangential and sagittal directions of the pupil.

        Parameters
        ----------
        
        Returns
        -------
        x : array_like
             X coordinates.
        y : array_like
             Y coordinates.
        """
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        xi = np.random.uniform(-size/2, size/2, (int(np.sqrt(N)), int(np.sqrt(N))))
        yi = np.random.uniform(-size/2, size/2, (int(np.sqrt(N)), int(np.sqrt(N))))

        indx = (xi**2 + yi**2 <= (size/2)**2)
        x, y = xi[indx], yi[indx]
        points = np.row_stack((x.flatten(), y.flatten()))
        del x, y
        x, y = points[0, :], points[1, :]
        
        return x, y

    def fromPupil(self, type='Polar'):
        """The function defining rays all over the pupil. This is uniformly random.

        Parameters
        ----------
        type : string
             'Polar' for a uniform polar distribution. This is default.
             'Grid' for a uniform grid distribution.
        
        Returns
        -------
        rays : batoid.RayVector
             Defining a uniform bundle of rays all over the pupil.
        """
        if type.casefold() == 'polar':
            x, y = self.createPolarBundleOfRays()
        elif type.casefold() == 'grid':
            x, y = self.createGridBundleOfRays()
        
        fieldX, fieldY = self.fieldX, self.fieldY
        rays = btd.RayVector.fromStop(x=x, y=y, 
                                      optic=self.opticalSys,
                                      wavelength=self.wavelength,
                                      dirCos=utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY)),
                                      flux=1.)
        return rays

    def fromPupilSingle(self, r):
        """The function defining rays all over the pupil. This is a user defined bundle.

        Parameters
        ----------
        r : list
             User defined bundle of ray(s). It can be one or several ones.
             For instance, r=[0, 0] will produce a single centered ray while
             r=[np.zeros(N), np.linspace(-R, R, N)] will produce a bundle of ray
             between -R and R on the Y axis.
        
        Returns
        -------
        rays : batoid.RayVector
             Defining a bundle of rays all over the pupil.
        """
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        
        fieldX, fieldY = self.fieldX, self.fieldY
        rays = btd.RayVector.fromStop(x=r[0], y=r[1], 
                                      optic=self.opticalSys,
                                      wavelength=self.wavelength,
                                      dirCos=utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY)),
                                      flux=1.)
        return rays
        
    def fromPupilForFan(self, axis='y'):
        """The function defining rays all over the pupil fro plotting a Ray Fan.

        Parameters
        ----------
        axis : string (len=1)
             Can only take 'x' and 'y' depeding if we want a uniform distribution over X or Y.
        
        Returns
        -------
        rays : batoid.RayVector
             Defining a bundle of rays all over the pupil.
        """
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        if axis.casefold() == 'y':
            x = np.zeros(int(N))
            y = np.linspace(-size, size, int(N)) / 2
        elif axis.casefold() == 'x':
            y = np.zeros(int(N))
            x = np.linspace(-size, size, int(N)) / 2
        
        fieldX, fieldY = self.fieldX, self.fieldY
        rays = btd.RayVector.fromStop(x=x, y=y, 
                                      optic=self.opticalSys,
                                      wavelength=self.wavelength,
                                      dirCos=utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY)),
                                      flux=1.)
        return rays

    def deleteVignettedRays(rv):
        """The function deleting vignetted rays (the ones being stopped by obscurations).
           Use rays being ".trace" not ".traceFull" or ".traceSplit".

        Parameters
        ----------
        rv : batoid.rayVector
             Bundle of rays to analyze and potentially delete.
        
        Returns
        -------
        rv : batoid.RayVector
             Keep unvignetted rays.
        """
        which = np.where(1-rv.vignetted)[0]
        #which = np.where(np.invert(rv.vignetted))[0].tolist()
        return rv[which]

    def intersection(self, rays):
        """The function computing the intersections between each surface.
           Use rays being ".traceFull" not ".trace" or ".traceSplit".

        Parameters
        ----------
        rays : batoid.RayVector
             Bundle of rays being trace at each surface meaning using ".traceFull".
        
        Returns
        -------
        intersec_dict : dict
             Dictionnary of array_like at each surface.
             For instance if surface 1 is called "S1", then we use ".get('S1')" to retrive 
             the array_like of all the rays with the surface ([x,y,z]).
        """
        optic = self.opticalSys
        listOfOpts = list(rays.keys())
        intersec_dict = dict()
        args = optic.items
        for i in range(len(listOfOpts)):
            optName = listOfOpts[i]
            
            r = rays.get(optName).get('out').r
            intersec_dict[optName] = btd.CoordTransform(args[i].coordSys, 
                                                        btd.CoordSys(origin=np.array([0, 0, 0]),
                                                                     rot=np.identity(3))).applyForwardArray(r[:, 0],
                                                                                                            r[:, 1],
                                                                                                            r[:, 2]).transpose()
        return intersec_dict

    def directionCos(self, rays):
        """The function computing the direction cosine vectors before and after each surface.
           Use rays being ".traceFull" not ".trace" or ".traceSplit".

        Parameters
        ----------
        rays : batoid.RayVector
             Bundle of rays being trace at each surface meaning using ".traceFull".
        
        Returns
        -------
        dir_dict : dict
             Dictionnary of array_like at each surface, before and after.
             For instance if surface 1 is called "S1", then we use ".get('S1').get('in')" to retrive 
             the array_like of all the directions before the surface. ".get('S1').get('out')" for the ones after.
        """
        optic, wl = self.opticalSys, self.wavelength
        listOfOpts = list(rays.keys())
        dir_dict = dict()
        args = optic.items
        for i in range(len(listOfOpts)):
            optName = listOfOpts[i]
            nIN, nOUT = args[i].inMedium.getN(wl), args[i].outMedium.getN(wl)
            
            vIN = rays.get(optName).get('in').v * nIN
            vOUT = rays.get(optName).get('out').v * nOUT
            dir_dict[optName] = {'in':vIN, 'out':vOUT}
        return dir_dict
    
class details:
    """Abstract base class defining different computations of the optical system.
    """
    def __init__(self, 
                 optic, fieldX=0, fieldY=0, wavelength=633e-9
                ):
        """
        Parameters
        ----------
        optic : batoid.Optic
            Optical system as a single optic or a compound optic.
        fieldX : float
            Field over X (in degrees). Default is ``0``.
        fieldY : float
            Field over Y (in degrees). Default is ``0``.
        wavelength : float
            Wavelength for this ray (in meters). Default is ``633e-9``.

        """
        self.optic = optic
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.wavelength = wavelength

    def opticalAxis(self):
        opt, wl = self.optic, self.wavelength
        rv = ray(opt, fieldX=0, fieldY=0, wavelength=wl, nbOfRays=1).fromPupilSingle([0, 0])
        opt.trace(rv)
        return rv.coordSys.toGlobal(rv.r)[0]

    def angleImgSpace(self):
        """The function computes the angle of the system in image space.

        Parameters
        ----------
        
        Returns
        -------
        theta_p : float
             Angle theta' in image space. The object is defined as infinity.
        """
        opt, wl = self.optic, self.wavelength
        D = opt.pupilSize
        n = opt.items[-1].inMedium.getN(wl)
        ray = btd.RayVector.fromStop(x=0, y=D/2,
                                     optic=opt,
                                     wavelength=wl,
                                     dirCos=utils.zemaxToDirCos(np.deg2rad(0), np.deg2rad(0)),
                                     flux=1.)

        opt.trace(ray, reverse=False)
        return np.arccos(ray.vz[0] * n)

    def paraxialWorkingFNumber(self):
        """The function computes the F number of the system in image space.

        Parameters
        ----------
        
        Returns
        -------
        f/# : float
             Paraxial working F number as being 1 / (2 * NA)
        """
        NA = self.NA()
        return 1 / (2 * NA)

    def NA(self):
        """The function computes the numerical aperture of the system in image space.

        Parameters
        ----------
        
        Returns
        -------
        NA : float
             Numerical aperture as being n' * sin(theta'). The object is defined as infinity.
        """
        n = self.optic.items[-1].inMedium.getN(self.wavelength)
        
        return n * np.sin(self.angleImgSpace())

    def FNumber(self):
        """The function computes the F number of the system.

        Parameters
        ----------
        
        Returns
        -------
        f/# : float
             F number as being EFFL/Dep.
        """
        D = self.optic.pupilSize
        EFFL = self.EFFL()
        return np.abs(EFFL) / D
    
    def angleImgSpaceFrom(self, x, y, offFieldX, offFieldY):
        """The function computes the angle of the system in image space from a specific point
           and from a specific off-axis angle.

        Parameters
        ----------
        x : float
             X position in the pupil
        y : float
             Y position in the pupil
        offFieldX : float
             Off-axis angle in degree along X.
        offFieldY : float
             Off-axis angle in degree along Y.
        
        Returns
        -------
        theta_p : float
             Angle theta' in image space.
        """
        opt, wl = self.optic, self.wavelength
        D = opt.pupilSize
        n = opt.items[-1].inMedium.getN(wl)
        ray = btd.RayVector.fromStop(x=x, y=y,
                                     optic=opt,
                                     wavelength=wl,
                                     dirCos=utils.zemaxToDirCos(np.deg2rad(offFieldX), np.deg2rad(offFieldY)),
                                     flux=1.)

        opt.trace(ray, reverse=False)
        return np.arccos(ray.vz[0] * n)

    def workingNA(self, offFieldX=0, offFieldY=0):
        """The function computes the 'working' numerical aperture of the system in image space.
           The useful image space angle is the difference between the one on the edge of the pupil
           angle and the off-axis angle T0.
           The useful NA defined as 'working' NA is the standard deviation of left/right/top/bottom.
           This is how it is defined in ZEMAX.

        Parameters
        ----------
        offFieldX : float
             Off-axis angle in degree along X. Default is 0.
        offFieldY : float
             Off-axis angle in degree along Y. Default is 0.
        
        Returns
        -------
        NAw : float
             Numerical aperture as being n' * sin(theta'-theta'_off). The object is defined as infinity.
        """
        D = self.optic.pupilSize
        n = self.optic.items[-1].inMedium.getN(self.wavelength)
        
        T0x = self.angleImgSpaceFrom(0, 0, offFieldX, 0)
        T0y = self.angleImgSpaceFrom(0, 0, 0, offFieldY)
        
        if (offFieldX == 0) & (offFieldY == 0):
            NA = self.NA()
            return NA
        elif (offFieldX == 0) & (offFieldY != 0):
            NA1 = n * np.sin(self.angleImgSpaceFrom(0, D/2, 0, offFieldY) - T0y)
            NA2 = n * np.sin(self.angleImgSpaceFrom(0, -D/2, 0, offFieldY) - T0y)
            return np.sqrt(np.mean(np.array([NA1, NA2])**2))
        elif (offFieldX != 0) & (offFieldY == 0):
            NA1 = n * np.sin(self.angleImgSpaceFrom(D/2, 0, offFieldX, offFieldY) - T0x)
            NA2 = n * np.sin(self.angleImgSpaceFrom(-D/2, 0, offFieldX, offFieldY) - T0x)
            return np.sqrt(np.mean(np.array([NA1, NA2])**2))
        else:
            NA1 = n * np.sin(self.angleImgSpaceFrom(0, D/2, 0, offFieldY) - T0y)
            NA2 = n * np.sin(self.angleImgSpaceFrom(0, -D/2, 0, offFieldY) - T0y)
            NA3 = n * np.sin(self.angleImgSpaceFrom(D/2, 0, offFieldX, 0) - T0x)
            NA4 = n * np.sin(self.angleImgSpaceFrom(-D/2, 0, offFieldX, 0) - T0x)
            return np.sqrt(np.mean(np.array([NA1, NA2, NA3, NA4])**2))

    def workingFNumber(self, offFieldX=0, offFieldY=0):
        """The function computes the working F number of the system.

        Parameters
        ----------
        offFieldX : float
             Off-axis angle in degree along X. Default is 0.
        offFieldY : float
             Off-axis angle in degree along Y. Default is 0.
        
        Returns
        -------
        f/#w : float
             Working F number as being 1 / (2 * NAw)
        """
        return 1 / (2 * self.workingNA(offFieldX, offFieldY))

    def airyRadius(self, offFieldX=0, offFieldY=0):
        """The function computes the Airy Radius (diffraction limit).

        Parameters
        ----------
        
        Returns
        -------
        Ra : float
             Airy Radius as being 1.22 * WL * Fw/# in microns.
             Note that 3.8317059 / pi is closer to the real value 
             of the first zero of the sombrero function than 1.22.
        """
        return (3.8317059/np.pi) * self.wavelength * self.workingFNumber(offFieldX, offFieldY) * 1e6

    def wavefront(self, npx=64, background=False):
        """The function computes the wavefront error in image space with respect to reference sphere.

        Parameters
        ----------
        npx : int
             Size of the grid which corresponds to the number of pixels. Default is ``64`` meaning 64x64.
        background : bool
             If False then the output array is a numpy.ma.MaskedArray (no data outside of the pupil).
             If True then the output array is a numpy.array (0 where there is no data).
        
        Returns
        -------
        WF : array_like, shape (npx,npx)
             Wavefront error with respect to reference sphere.
             Note that the projection is set as ``zemax`` and reference as ``chief``.
             Plus the function returns minus the wavefront because of the orientation
             definition.
        """
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        #Rref = btd.analysis.exitPupilPos(optic=opt, wavelength=wl)[-1] - opt.items[-1].coordSys.origin[-1]
        XPpos = utils.exitPupilPos(optic=opt, wavelength=wl)
        Rref = XPpos[-1] - opt.items[-1].coordSys.origin[-1]
        WF = utils.wavefront(optic=opt,
                             theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                             wavelength=wl,
                             nx=npx,
                             sphereRadius=Rref,
                             projection='zemax',
                             reference='chief')
        
        if background:
            return (WF.array.data) * ~WF.array.mask
        else:
            return WF.array

    def wavefrontZnk(self, npx=64, jmax=12, eps=0, show=False):
        """The function computes Zernikes coefficients of the wavefront using Noll's definition.

        Parameters
        ----------
        npx : int
             Size of the grid which corresponds to the number of pixels. Default is ``64`` meaning 64x64.
        jmax : int
             Maximum Zernike coeficient. Default is ``12``.
        show : bool
             Displaying coefficients like in Zemax. Default is ``False``.
             Note that, for the moment, diplaying works for jmax <= 14.
        
        Returns
        -------
        Znk : array_like, len (jmax+1)
             Zernike coefficient in waves.
             Note that the first element is 0 so that Znk[i] corresponds to the i-th 
             Zernike coefficient as defined by Noll.
        """
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        #Rref = btd.analysis.exitPupilPos(optic=opt, wavelength=wl)[-1] - opt.items[-1].coordSys.origin[-1]
        XPpos = utils.exitPupilPos(optic=opt, wavelength=wl)
        Rref = XPpos[-1] - opt.items[-1].coordSys.origin[-1]
        #Znk = btd.analysis.zernikeGQ(optic=opt,
         #                            theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
          #                           wavelength=wl,
           #                          rings=npx,
            #                         sphereRadius=Rref, 
             #                        projection='zemax',
              #                       reference='chief', 
               #                      jmax=jmax, eps=0.2)
        #Znk = btd.analysis.zernike(optic=opt,
         #                          theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
          #                         wavelength=wl,
           #                        projection='zemax',
            #                       nx=npx,
             #                      sphereRadius=Rref,
              #                     reference='chief',
               #                    jmax=jmax, eps=0.2)
        Znk = utils.zernike(optic=opt,
                            theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                            wavelength=wl,
                            projection='zemax',
                            nx=npx,
                            sphereRadius=Rref,
                            reference='chief',
                            jmax=jmax, eps=eps)
        
        ## The orientation is diffrent to Zemax and so are how angles are defined.
        ## All spherical coefficients need to be multiplied by -1 + the angle is negative so sines as to be remultiplied by -1
        #Znk *= -1
        #val = np.linspace(0, len(Znk)-1, len(Znk), dtype='int') # -1 where SIN => negative angle
        #mask = np.logical_and.reduce((val%2==1, val>1, val!=11, val!=22, val!=37))
        #Znk[mask] *= -1

        if show:
            Noll_Znk_names = ["", "1", "4^(1/2) (P) * COS (A)", "4^(1/2) (P) * SIN (A)", "3^(1/2) (2P^2-1)", 
                              "6^(1/2) (p^2) * SIN (2A)", "6^(1/2) (p^2) * COS (2A)", "8^(1/2) (3p^3 - 2p) * SIN (A)",
                              "8^(1/2) (3p^3 - 2p) * COS (A)", "8^(1/2) (p^3) * SIN (3A)", "8^(1/2) (p^3) * COS (3A)", 
                              "5^(1/2) (6p^4-6p^2+1)", "10^(1/2) (4p^4-3p^2) * COS (2A)", "10^(1/2) (4p^4 - 3p^2) * SIN (2A)"]
            print(f'RMS (to chief)    :    {Znk[np.abs(Znk) > 1e-7].std():.9f} (waves)\n')
            for i in range(1, len(Znk)):
                if i<=9:
                    if Znk[i] >= 0:
                        print(f'{"Z":<4}{i:<7}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}')
                    else:
                        print(f'{"Z":<4}{i:<6}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}')
                else:
                    if Znk[i] >= 0:
                        print(f'{"Z":<3}{i:<8}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}') 
                    else:
                        print(f'{"Z":<3}{i:<7}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}')
        return Znk

    def EFFL(self):
        """The function computes the Effective Focal Length (EFFL) of the system.

        Parameters
        ----------
        
        Returns
        -------
        EFFL : float
            EFFL in meters.
        """
        return btd.analysis.focalLength(optic=self.optic,
                                        theta_x=0, theta_y=0,
                                        wavelength=self.wavelength,
                                        projection='zemax')

    def posChiefRayAtImgPlane(self):
        """The function computes the position of the chief ray in the image plane.

        Parameters
        ----------
        
        Returns
        -------
        x, y : float, float
            Position of the chief ray in the image plane.
        """
        anglX, anglY = self.fieldX*np.pi/180, self.fieldY*np.pi/180
        chiefRay = btd.RayVector.fromStop(x=0, y=0,
                                          optic=self.optic,
                                          wavelength=self.wavelength,
                                          dirCos=utils.zemaxToDirCos(np.deg2rad(anglX), np.deg2rad(anglY)),
                                          flux=1.)
        self.optic.trace(chiefRay, reverse=False)
        return chiefRay.r[0]

    def strehlRatio(self):
        """The function computes the Strehl ratio of the system.
           The tilt free wavefront is along X and Y.

        Parameters
        ----------
        
        Returns
        -------
        strehlRation : float
            Strehl ration as being exp(-(2*WFrms)^2).
        """
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY

        dt = details(opt, fieldX=fieldX, fieldY=fieldY, wavelength=wl)
        WF = dt.wavefront(npx=1024)
        #corrWF, _ = correction.removeTiltWavefrontBrut(WF, wl, tiltX=False, tiltY=True)
        Znk = dt.wavefrontZnk(npx=128, jmax=37, show=False)
        corrWF, _ = correction.removeTiltWavefront(WF, Znk, removeTilt=True, removePiston=False)
        RMS = corrWF.std()
        return np.exp(-(2*np.pi*RMS)**2)

    def huygensPSF(self, npx=32):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        
        StrehlRatio = self.strehlRatio()
        
        PSF = btd.analysis.huygensPSF(optic=opt,
                                      wavelength=wl,
                                      theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                                      nx=npx,
                                      nxOut=npx,
                                      projection='zemax',
                                      reference='chief')
        PSF = PSF.array * StrehlRatio / np.max(PSF.array)
        return PSF, StrehlRatio

    def fftPSF(self, npx=16, pad=2):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY

        StrehlRatio = self.strehlRatio()

        Rref = btd.analysis.exitPupilPos(optic=opt, wavelength=wl)[-1] - opt.items[-1].coordSys.origin[-1]
        
        PSF = btd.analysis.fftPSF(optic=opt,
                                     wavelength=wl,
                                     theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                                     nx=npx,
                                     pad_factor=pad,
                                     sphereRadius=Rref,
                                     projection='zemax',
                                     reference='chief')
        PSF = PSF.array * StrehlRatio / np.max(PSF.array)
        return PSF, StrehlRatio

class correction:

    def removeTiltWavefrontBrut(WF, wl, tiltX=True, tiltY=True):
        if isinstance(WF, btd.lattice.Lattice):
            WF = WF.array
            
        rows, cols = WF.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x, y = np.meshgrid(x, y)

        X = np.vstack((x.flatten(), y.flatten())).T
        Z = WF.flatten()

        coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
        a, b = coeffs
        if tiltX and tiltY:
            fitted_plane = a * x + b * y
            removed = np.array([a, b]) * -wl * 1e9
        elif tiltX and not tiltY:
            fitted_plane = a * x
            removed = np.array([a, 0.0]) * -wl * 1e9
        elif not tiltX and tiltY:
            fitted_plane = b * y
            removed = np.array([0.0, b]) * -wl * 1e9

        corrWF = WF - fitted_plane.reshape(rows, cols)
        avg = (np.max(corrWF) + np.min(corrWF)) / 2

        return corrWF - avg, removed

    def removeTiltWavefront(WF, Znk, removeTilt=True, removePiston=False):
        import galsim

        removed = []
        if removePiston:
            Piston = Znk[1]
            removed.append(Piston)
            Znk[1] = 0.0
        if removeTilt:
            X_tilt, Y_tilt = 2*Znk[2], 2*Znk[3]
            removed.append(X_tilt)
            removed.append(Y_tilt)
            Znk[2], Znk[3] = 0.0, 0.0
    
        size = np.shape(WF)
        
        val = np.linspace(-1, 1, size[0])
        x, y = np.meshgrid(val, val)

        x = np.ma.masked_array(data=x, mask=WF.mask)
        y = np.ma.masked_array(data=y, mask=WF.mask)
        
        W = galsim.zernike.Zernike(coef=Znk).evalCartesian(x, y)
        W = np.ma.masked_array(data=W, mask=WF.mask)

        return W, removed
    
class plot:

    def sag3D(optic):
        inDiam, outDiam = optic.inDiam, optic.outDiam
        th = 2 * np.pi * np.random.rand(int(1e6))
        cth, sth = np.cos(th), np.sin(th)
        d = (outDiam - inDiam) * np.random.rand(int(1e6)) + inDiam
        x = (d/2) * cth
        y = (d/2) * sth
        z = optic.surface.sag(x, y) * 1000
        
        xi = np.linspace(x.min(), x.max(), 64)
        yi = np.linspace(y.min(), y.max(), 64)
        X,Y = np.meshgrid(xi, yi)
        Z = griddata((x,y), z, (X,Y), method='nearest')
        
        fig = go.Figure(data=[go.Surface(x=xi,y=yi,z=Z,
                        colorscale='Spectral',
                        reversescale=True,
                        colorbar=dict(thickness=30,
                                      tickvals=np.linspace(np.min(Z), np.max(Z), 10),
                                      title='Surface Sag (mm)'))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          scene=dict(zaxis=dict(range=[np.min(Z), 0],title='sag (mm)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')))
        fig.show()

    def sag3D_TEST(surface, outDiam):
        inDiam = 0
        th = 2 * np.pi * np.random.rand(int(1e6))
        cth, sth = np.cos(th), np.sin(th)
        d = (outDiam - inDiam) * np.random.rand(int(1e6)) + inDiam
        x = (d/2) * cth
        y = (d/2) * sth
        z = surface.sag(x, y)
        
        xi = np.linspace(x.min(), x.max(), 64)
        yi = np.linspace(y.min(), y.max(), 64)
        X,Y = np.meshgrid(xi, yi)
        Z = griddata((x,y), z, (X,Y), method='nearest')
        
        fig = go.Figure(data=[go.Surface(x=xi,y=yi,z=Z,
                        colorscale='Spectral',
                        reversescale=True,
                        colorbar=dict(thickness=30,
                                      tickvals=np.linspace(np.min(Z), np.max(Z), 10),
                                      title='Surface Sag (waves)'))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          scene=dict(zaxis=dict(title='sag (waves)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')))
        fig.show()
        return xi, yi, Z     
    def sag3D_TEST_residual(xi, yi, Z):  
        fig = go.Figure(data=[go.Surface(x=xi,y=yi,z=Z,
                        colorscale='Spectral',
                        reversescale=True,
                        colorbar=dict(thickness=30,
                                      tickvals=np.linspace(np.min(Z), np.max(Z), 10),
                                      title='Surface Sag (waves)'))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          scene=dict(zaxis=dict(title='residual sag (waves)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')))
        fig.show()

    def spotDiagram(rvObj, fieldBias, rvImg, wl, airyR=None, scale=None):
        ## Centroid method
        cx = np.sum(rvImg.x)/len(rvImg.x)
        cy = np.sum(rvImg.y)/len(rvImg.y)

        fieldsObj = rvObj.v[0]
        anglX, anglY = np.round(np.rad2deg(fieldsObj[0]), decimals=3), np.round(np.rad2deg(fieldsObj[1])-fieldBias, decimals=3)

        fig, ax = plt.subplots()
        ax.scatter((rvImg.x - cx) * 1e6, (rvImg.y - cy) * 1e6, s=0.25 ,c='blue', marker='.')
        
        if airyR is not None:
            ax.add_patch(plt.Circle((0, 0), airyR, color='k', fill=False))
        ax.set_title(f'OBJ: {anglX}, {anglY} (deg)', fontsize=14)
        ax.set_xlabel(f'IMA: {np.round(cx * 1000, decimals=3)}, {np.round(cy * 1000, decimals=3)} mm', fontsize=14)
        if scale is not None:
            ax.set_xlim(-scale/2, scale/2)
            ax.set_ylim(-scale/2, scale/2)
            ax.set_ylabel(f'{float(scale)}', fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both',       # changes apply to the both axis
                       which='both',      # both major and minor ticks are affected
                       bottom=False,
                       top=False,
                       left=False,
                       right=False,
                       labelbottom=False,
                       labelleft=False)
        ax.grid()
        ax.set_axisbelow(True)
        fig.canvas.header_visible = False
        plt.subplots_adjust(left=-0.1)
        plt.show()

        print(f'{date.today().strftime("%m/%d/%Y")}')
        if airyR is not None:
            print(f'Units are µm. Airy Radius: {np.round(airyR, decimals=3)} µm. Wavelength: {wl * 1e6} µm.')
        Rms_R = np.sqrt(np.sum((rvImg.x - cx)**2)/len(rvImg.x) + np.sum((rvImg.y - cy)**2)/len(rvImg.y)) * 1e6
        print(f'RMS radius :    {np.round(Rms_R, decimals=3)}')
        R = np.sqrt((rvImg.x - cx)**2 + (rvImg.y - cy)**2)
        Geo_R = np.max(R) * 1e6
        print(f'GEO radius :    {np.round(Geo_R, decimals=3)}')
        if scale is not None:
            print(f'Scale bar  :   {float(scale)}    Reference : Centroid\n')

    def wavefront(WF, wl, fields, title='', rotation='0', XPdiam=0, tiltrmv=None):
        if isinstance(WF, btd.lattice.Lattice):
            WF = WF.array
        else:
            pass

        if rotation == '90':
            WF = np.rot90(WF)
        elif rotation == '180':
            WF = np.flipud(WF)
        elif rotation == '270':
            WF = np.flipud(WF)
            WF = np.rot90(WF)
        else:
            pass
        
        fig, ax = plt.subplots()
        WFax = ax.imshow(WF, cmap=plot.customDivergingColormap(), extent=[-1., 1., -1., 1.])
        ax.set_facecolor('#2822bb')
        #ax.set_title('Wavefront Map', fontsize=14)
        ax.set_xlabel('X-Pupil (Rel. Units)', fontsize=14)
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_ylabel('Y-Pupil (Rel. Units)', fontsize=14)
        ax.set_yticks([-1.0, 0.0, 1.0])
        cbar = fig.colorbar(WFax, label='waves')
        cbar.set_ticks(np.round(np.linspace(WF.min(), WF.max(), 11), decimals=4))
        #WFax.set_clim(-5, 5)
        #cbar.set_ticks(np.round(np.linspace(-5, 5, 11), decimals=4))
        fig.canvas.header_visible = False
        ax.set_title(title)
        plt.show()

        print(f'{date.today().strftime("%m/%d/%Y")}')
        print(f'{float(wl * 1e6)} µm at {float(fields[0])}, {float(fields[1])} (deg)')
        degsymb = '\u00b0'
        print(f'Rotation: {rotation}{degsymb}')
        RMS = WF.std()
        PV = np.abs(WF.max()-WF.min())
        print('Peak to valley =', np.round(PV, decimals=4), 'waves, RMS =', np.round(RMS, decimals=4), 'waves')
        if XPdiam != 0:
            print('Exit Pupil Diameter:', "{:e}".format(XPdiam * 1000), 'Millimeters')
        if tiltrmv is not None:
            if len(tiltrmv) == 0:
                pass
            elif len(tiltrmv) == 1:
                print(f'Piston Removed : {np.round(tiltrmv[0], decimals=4)} waves')
            elif len(tiltrmv) == 3:
                print(f'Piston Removed : {np.round(tiltrmv[0], decimals=4)} waves, Tilt Removed:  X = {np.round(tiltrmv[1], decimals=4)}, Y = {np.round(tiltrmv[2], decimals=4)} waves\n')
            else:
                print(f'Tilt Removed:  X = {np.round(tiltrmv[0], decimals=4)}, Y = {np.round(tiltrmv[1], decimals=4)} waves\n')
            

    def customDivergingColormap():
        nbOfBits = 512
        N = nbOfBits//7
        s = 1.2
        #jet_colors = plt.cm.get_cmap('jet')
        jet_colors = mpl.colormaps.get_cmap('jet')
        lower_colors = jet_colors(np.linspace(0.10, 1/3, int(s*N)))
        upper_colors = jet_colors(np.linspace(2/3, 0.95, int(1.5*s*N)))
        middle_colors = jet_colors(np.linspace(1/3, 2/3, int(nbOfBits-2.5*s*N)))
        custom_colors = np.vstack((lower_colors, middle_colors, upper_colors))
        return mcolors.ListedColormap(custom_colors)

    def findPxSize(PSF, airyR):
        N = len(PSF[0])
        mid = N//2
        CS = PSF[mid, :]
        minima = argrelextrema(CS, np.less)[0]
        indx = np.searchsorted(minima, mid)
        indx = minima[indx]
        pxSize = airyR/(indx-mid+1)
        return pxSize
        

    def psf3D(PSF, airyR, thresh=0):
        #PSF = np.log(PSF.array)
        #plot.thresholding(PSF, thresh=thresh)
        N = len(PSF[0])
        mid = N//2
        pxSize = plot.findPxSize(PSF, airyR)
        x = np.linspace(-mid*pxSize, mid*pxSize, N)
        x, y = np.meshgrid(x, x)
        
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=PSF,
                                         colorscale='Spectral',
                                         reversescale=True,
                                         colorbar=dict(thickness=30,
                                                       tickvals=np.round(np.linspace(PSF.min(), PSF.max(), 10), decimals=1)
                                                      )
                                        )
                             ]
                       )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          width=700,
                          height=500,
                          xaxis_range=[-airyR*6.557, airyR*6.557],
                          yaxis_range=[-airyR*6.557, airyR*6.557]
                         )
        fig.show()

    def _forward(x):
        return 10**x
    def _inverse(x):
        return np.log(x)
    
    def psf2D(PSF, airyR, thresh=0):
        #PSF = np.log(PSF)
        #plot.thresholding(PSF, thresh=thresh)
        N = len(PSF[0])
        mid = N//2
        pxSize = plot.findPxSize(PSF, airyR)
        
        fig, ax = plt.subplots()
        PSFax = ax.imshow(PSF, cmap='gray', norm=mcolors.LogNorm(vmin=1e-5, vmax=1), extent=[-mid*pxSize,mid*pxSize,-mid*pxSize,mid*pxSize])
        ax.set_title(f'PSF', fontsize=14)
        ax.set_xlabel('µm', fontsize=14)
        ax.set_ylabel('µm', fontsize=14)
        ax.set_xlim(-airyR*6.557, airyR*6.557)
        ax.set_ylim(-airyR*6.557, airyR*6.557)
        cbar = fig.colorbar(PSFax)
        cbar.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        fig.canvas.header_visible = False
        plt.show()

    def psfCrossSection(PSF, airyR, wl=None, fields=None, stehlRatio=None, axis='Y'):
        #PSF = np.sqrt(PSF)

        N = len(PSF[0])
        mid = N//2
        
        fig, ax = plt.subplots()
        if axis.casefold() == 'y':
            CS = PSF[:, mid]
            ax.set_title(f'Cross Section of PSF at Y=0', fontsize=14)

            pxSize = plot.findPxSize(PSF, airyR)
        elif axis.casefold() == 'x':
            CS = PSF[mid, :]
            ax.set_title(f'Cross Section of PSF at X=0', fontsize=14)

            pxSize = plot.findPxSize(PSF, airyR)

        x = np.linspace(-mid*pxSize, mid*pxSize, N)
        x -= x[CS.tolist().index(np.max(CS))]
        ax.plot(x, CS)
        ax.set_xlabel('µm', fontsize=14)
        ax.set_ylabel('Relative Irradiance', fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xlim(-airyR*6.557, airyR*6.557)
        ax.set_yticks(np.arange(0, 1, 0.1))
        fig.canvas.header_visible = False
        plt.grid(which='major')
        plt.grid(which='minor', color='#EEEEEE', linestyle=':')
        ax.minorticks_on()
        plt.show()

        print(f'{date.today().strftime("%m/%d/%Y")}')
        if (wl is not None) and (fields is not None):
            print(f'{float(wl * 1e6)} µm at {float(fields[0])}, {float(fields[1])} (deg)')
        print(f'Image width is {2*airyR*6.557:.2f} µm')
        if stehlRatio is not None:
            print(f'Strehl ratio: {stehlRatio:.3f}')

    def thresholding(data, thresh=0):
        if thresh != 0:
            lim = thresh * np.max(data)
            data[data<lim] = 0
        return data

    def rayFan(opticalSys, fieldX=0, fieldY=0, wavelength=633e-9, axis='y', show=True):
        ## Incoming ray
        nbOfRays = 1e3
        rays0 = ray(opticalSys, fieldX=fieldX, fieldY=fieldY, wavelength=wavelength, nbOfRays=nbOfRays).fromPupilForFan(axis=axis)
        rays = rays0.copy()
        
        opticalSys.trace(rays, reverse=False)
        rays = ray.deleteVignettedRays(rays)

        ## Coordinates on Detector
        if axis == 'y':
            c = np.sum(rays.y)/len(rays.y)
            e = (rays.y - c) * 1e3
        elif axis == 'x':
            c = np.sum(rays.x)/len(rays.x)
            e = (rays.x - c) * 1e3

        ## Detector at Exit Pupil
        XPpos = btd.analysis.exitPupilPos(optic=opticalSys, wavelength=wavelength)[-1]
        Dcoord = defineCoordinate(posZ=XPpos).local()
        originalD = opticalSys.items[-1]
        D = optic('D', inDiam=originalD.inDiam, outDiam=originalD.outDiam, coordSys=Dcoord).detector()
        newSysAtXP = build.compoundOptic(*opticalSys.items[0:-1], D,
                                         pupilSize=opticalSys.pupilSize, backDist=opticalSys.backDist,
                                         EPcoord=opticalSys.stopSurface.coordSys)

        newSysAtXP.trace(rays0, reverse=False)
        rays0 = ray.deleteVignettedRays(rays0)

        ## Coordinates in Exit Pupil
        if axis == 'y':
            P = rays0.y / np.max(np.abs(rays0.y))
        elif axis == 'x':
            P = rays0.x / np.max(np.abs(rays0.x))
        
        ## Plot P=f(e)
        if show:
            fig, ax = plt.subplots()
            ax.plot(P, e, linestyle="-", marker=".", markersize=5, zorder=2)
            ybound = ax.get_ylim()
            ax.plot([0, 0], [ybound[0], ybound[1]], color='k', zorder=1)
            ax.plot([-1, 1], [0, 0], color='k', zorder=0)
            ax.grid()
            ax.set_xlim(-1, 1)
            ax.set_ylim(ybound[0], ybound[1])
            #plt.ylim(-0.01, 0.01)
            ax.set_title(f'Ray Fan on {axis.upper()}-axis for field of {fieldX}°, {fieldY}°')
            ax.set_ylabel(f'e{axis} (mm)')
            ax.set_xlabel(f'P{axis} (normalized)')
            plt.subplots_adjust(left=+0.17)
            plt.show()
        return e, P
        
class layout:

    def sag2DY(optic, inDiam, outDiam):
        N = 0.5e2
        if inDiam != 0:
            y1 = np.linspace(-outDiam, -inDiam, int(N/2))/2
            x1 = np.zeros(len(y1))
            z1 = optic.surface.sag(x1, y1)
            x1, y1, z1 = btd.CoordTransform(optic.coordSys, 
                                            btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x1, y1, z1)
            y2 = np.linspace(inDiam, outDiam, int(N/2))/2
            x2 = np.zeros(len(y2))
            z2 = optic.surface.sag(x2, y2)
            x2, y2, z2 = btd.CoordTransform(optic.coordSys, 
                                            btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x2, y2, z2)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            y = np.linspace(-outDiam, outDiam, int(N))/2
            x = np.zeros(len(y))
            z = optic.surface.sag(x, y)
            x, y, z = btd.CoordTransform(optic.coordSys,
                                         btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x, y, z)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df
        
    def sag2DX(optic, inDiam, outDiam):
        N = 0.5e2
        if inDiam != 0:
            x1 = np.linspace(-outDiam, -inDiam, int(N/2))/2
            y1 = np.zeros(len(x1))
            z1 = optic.surface.sag(x1, y1)
            x1, y1, z1 = btd.CoordTransform(optic.coordSys, 
                                            btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x1, y1, z1)
            x2 = np.linspace(inDiam, outDiam, int(N/2))/2
            y2 = np.zeros(len(x2))
            z2 = optic.surface.sag(x2, y2)
            x2, y2, z2 = btd.CoordTransform(optic.coordSys, 
                                            btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x2, y2, z2)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            x = np.linspace(-outDiam, outDiam, int(N))/2
            y = np.zeros(len(x))
            z = optic.surface.sag(x, y)
            x, y, z = btd.CoordTransform(optic.coordSys,
                                         btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x, y, z)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df 

    def sag2Dcircle(optic, inDiam, outDiam):
        N = 0.5e2
        if inDiam != 0:
            th = np.linspace(0, 2*np.pi, int(N))
            x1 = outDiam/2 * np.cos(th)
            y1 = outDiam/2 * np.sin(th)
            z1 = optic.surface.sag(x1, y1)
            x1, y1, z1 = btd.CoordTransform(optic.coordSys, 
                                            btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x1, y1, z1)
            x2 = inDiam/2 * np.cos(th)
            y2 = inDiam/2 * np.sin(th)
            z2 = optic.surface.sag(x2, y2)
            x2, y2, z2 = btd.CoordTransform(optic.coordSys, 
                                            btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x2, y2, z2)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            th = np.linspace(0, 2*np.pi, int(N))
            x = outDiam/2 * np.cos(th)
            y = outDiam/2 * np.sin(th)
            z = optic.surface.sag(x, y)
            x, y, z = btd.CoordTransform(optic.coordSys,
                                         btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x, y, z)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df

    def getMirrorLines(opt, inner, outter):
        if inner != 0:
            df1Y, df2Y = layout.sag2DY(opt, inner, outter)
            df1X, df2X = layout.sag2DX(opt, inner, outter)
            df1c, df2c = layout.sag2Dcircle(opt, inner, outter)
            return df1Y, df2Y, df1X, df2X, df1c, df2c
        else: 
            dfY = layout.sag2DY(opt, 0, outter)
            dfX = layout.sag2DX(opt, 0, outter)
            dfc = layout.sag2Dcircle(opt, 0, outter)
            return dfY, dfX, dfc

    def getAllLines(optic):
        N = len(optic.items)
        allLines = [0]* N
        for i in range(N):
            opt = optic.items[i]
            allLines[i] = layout.getMirrorLines(opt, opt.inDiam, opt.outDiam)
        return chain(*allLines)

    def rayBundle(optic, WL, inDiam, outDiam, anglX, anglY, nbOfRays):
        if inDiam != 0:
            y = np.concatenate((np.linspace(-outDiam, -inDiam, int(nbOfRays/2)), np.linspace(inDiam, outDiam, int(nbOfRays/2)))) / 2
        else:
            y = np.linspace(-outDiam, outDiam, int(nbOfRays)) / 2
        rays = [0]*len(y)
        for i in range(len(y)):
            rays[i] = layout.getRay(optic, WL=WL, x=0, y=y[i], anglX=anglX, anglY=anglY)
        return rays

    def getRay(optic, WL, x=0, y=0, anglX=0, anglY=0):
        args = optic.items
        ray = btd.RayVector.fromStop(x=x, y=y,
                                     optic=optic,
                                     wavelength=WL,
                                     dirCos=utils.zemaxToDirCos(np.deg2rad(anglX), np.deg2rad(anglY)),
                                     flux=1.)
        ray = optic.traceFull(ray, reverse=False)
        listOfOpts = list(ray.keys())
        wholeRayTracing = [0] * len(args)
        for i in range(len(listOfOpts)):
            optName = listOfOpts[i]
            rIN, rOUT = ray.get(optName).get('in').r[0], ray.get(optName).get('out').r[0]
            coordSys = args[i].coordSys
            if i == 0:
                x, y, z = btd.CoordTransform(coordSys,
                                             btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(rOUT[0], rOUT[1], rOUT[2])
                wholeRayTracing[i] = np.array([[rIN[0], x], [rIN[1], y], [rIN[2], z]])
            else:
                coordSysp = args[i-1].coordSys
                r = btd.CoordTransform(coordSys,
                                       btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(rOUT[0],
                                                                                                                       rOUT[1],
                                                                                                                       rOUT[2])
                rp = btd.CoordTransform(coordSysp,
                                        btd.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(rIN[0],
                                                                                                                        rIN[1],
                                                                                                                        rIN[2])
                wholeRayTracing[i] = np.array([rp, r]).transpose()
        return wholeRayTracing

    def visualized2D(nbOfRaysPerField, *args):
        colors = np.array([['indigo'], ['blue'], ['green'], ['yellow'], ['orange'], ['red']])
        c, i = 0, 0
        fig = px.line(width=1000, height=700)
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                fig.add_trace(px.line(arg, x='z', y='y', color_discrete_sequence=['black'],).data[0])
            else:
                for j in range(0, len(arg)): # start at 1 => no ray from inf
                    if c > len(colors):
                        c = 0
                    line = arg[j]
                    df = pd.DataFrame({"x":line[0], "y":line[1], "z":line[2]})
                    if (i < nbOfRaysPerField) or (nbOfRaysPerField == 0):
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[0]).data[0]) #colors[c]
                    elif ((i >= nbOfRaysPerField) and (i < 2*nbOfRaysPerField)) and (nbOfRaysPerField != 0):
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[2]).data[0]) #colors[c]
                    else:
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[-1]).data[0]) #colors[c]
                c += 1
                i += 1
        
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          scene=dict(xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')),
                          paper_bgcolor='rgb(255,255,255)',
                          plot_bgcolor='rgb(255,255,255)')
        fig.show()

    def visualized3D(nbOfRaysPerField, *args):
        colors = np.array([['indigo'], ['blue'], ['green'], ['yellow'], ['orange'], ['red']])
        c, i = 0, 0
        fig = px.line_3d(width=1000, height=700)
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                fig.add_trace(px.line_3d(arg, x='x', y='y', z='z', color_discrete_sequence=['black'],).data[0])
            else:
                for j in range(0, len(arg)): # start at 1 => no ray from inf
                    if c > len(colors):
                        c = 0
                    line = arg[j]
                    df = pd.DataFrame({"x":line[0], "y":line[1], "z":line[2]})
                    if (i < nbOfRaysPerField) or (nbOfRaysPerField == 0):
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[0]).data[0]) #colors[c]
                    elif ((i >= nbOfRaysPerField) and (i < 2*nbOfRaysPerField)) and (nbOfRaysPerField != 0):
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[2]).data[0]) #colors[c]
                    else:
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[-1]).data[0]) #colors[c]
                c += 1
                i += 1
        camera_params = dict(up=dict(x=0,y=1,z=0),
                             center=dict(x=0,y=0,z=0),
                             eye=dict(x=-1.5,y=0,z=0.455))
    
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          scene=dict(zaxis=dict(title='z (meters)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')),
                          scene_camera=camera_params)
        fig.show()