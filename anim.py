import numpy as np
from numpy import sin,cos,pi,sqrt # makes the code more readable
import pylab as plt
from mayavi import mlab # or from enthought.mayavi import mlab
from scipy.optimize import newton

def roche(r,theta,phi,pot,q):
    lamr,nu = r*cos(phi)*sin(theta),cos(theta)
    return (pot - (1./r  + q*( 1./sqrt(1. - 2*lamr + r**2) - lamr)  + 0.5*(q+1) * r**2 * (1-nu**2) ))

theta,phi = np.mgrid[0:np.pi:75j,-0.5*pi:1.5*np.pi:150j]