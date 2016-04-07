from process.BaseProcess import BaseProcess, SliderLabel
import pyqtgraph as pg
from PyQt4.QtGui import *
from PyQt4.QtCore import Qt
import global_vars as g
from leastsqbound import leastsqbound
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter as gf
from window import Window
from roi import ROI_rectangle
import scipy


class Drift_Correction(BaseProcess):
    '''Draw rectangular ROIs around tracer locations to track movement over time. Locate the center coordinates and correct the image for drift
    '''
    def __init__(self):
        super().__init__()
        findButton = QPushButton("Find Centers")
        findButton.pressed.connect(self.find_centers)
        self.slider = SliderLabel()
        self.slider.setRange(0, 10)
        self.slider.valueChanged.connect(self.plotCenters)
        self.items.append({'name':'findButton','string':'Locate centroids','object':findButton})
        self.items.append({'name':'smoothness', 'string': 'Drift Smoothness', 'object':self.slider})

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)
        self.scatter = pg.ScatterPlotItem()
        
        self.p1 = pg.PlotWidget(title="X Shift")
        self.p2 = pg.PlotWidget(title="Y Shift")

    def gui(self):
        super().gui()
        self.ui.layout.insertWidget(0, self.p1)
        self.ui.layout.insertWidget(1, self.p2)
        self.ui.resize(1000, 600)

    def plotCenters(self):
        self.p1.clear()
        self.p2.clear()
        for r in self.rois:
            centers = np.copy(r['centers'])
            #self.p1.plot(y=centers[:, 0] / np.average(centers[:, 0]), pen=r['roi'].pen)
            #self.p2.plot(y=centers[:, 1] / np.average(centers[:, 1]), pen=r['roi'].pen)
            self.p1.plot(y=gf(centers[:, 0], self.slider.value()), pen=r['roi'].pen)
            self.p2.plot(y=gf(centers[:, 1], self.slider.value()), pen=r['roi'].pen)

    def __call__(self, keepSourceWindow=False):
        xx, yy = [], []
        for i in range(len(self.p1.plotItem.items)):
            xx.append(self.p1.plotItem.items[i].getData()[1])
            yy.append(self.p2.plotItem.items[i].getData()[1])
        diffs = np.mean([xx, yy], 1).T
        diffs = -1 * diffs + diffs[0]
        
        im = np.copy(g.m.currentWindow.image)
        for i, sh in enumerate(diffs):
            g.m.statusBar().showMessage("shifting frame %d of %d" % (i, len(diffs)))
            im[i] = shift(im[i], sh)
            QApplication.instance().processEvents()
        return Window(im)


    def find_centers(self):
        win = g.m.currentWindow
        im = win.image
        mx,my=win.imageDimensions()
        self.rois = []
        g.centers = []
        for roi in g.m.currentWindow.rois:
            mask = roi.mask
            mask=mask[(mask[:,0]>=0)*(mask[:,0]<mx)*(mask[:,1]>=0)*(mask[:,1]<my)]

            xx=mask[:,0]; yy=mask[:,1]
            centers = []

            for frame in im:
                gframe = gf(frame, 1)
                x0, y0 = np.unravel_index(gframe.argmax(), gframe.shape)
                #centers.append([x0, y0])
                #vals = fitGaussian(frame, (x0, y0, 1, 3))
                #x1, y1, a, b = vals[0]
                centers.append([x0, y0])
            self.rois.append({'roi': roi, 'centers': centers})
        self.plotCenters()

drift_correction = Drift_Correction()

def fitGaussian(I=None, p0=None, bounds=None):
    '''
    Takes an nxm matrix and returns an nxm matrix which is the gaussian fit
    of the first.  p0 is a list of parameters [xorigin, yorigin, sigma,amplitude]
    '''

    x=np.arange(I.shape[0])
    y=np.arange(I.shape[1])
    X=[x,y]
    p0=[round(p,3) for p in p0] 
    p, cov_x, infodic, mesg, ier = leastsqbound(err, p0,args=(I,X),bounds = bounds,ftol=.0000001,full_output=True)
    #xorigin,yorigin,sigmax,sigmay,angle,amplitude=p
    I_fit=gaussian(x[:,None], y[None,:],*p)
    return p, I_fit, I_fit
    
        
def gaussian(x,y,xorigin,yorigin,sigma,amplitude):
    '''xorigin,yorigin,sigmax,sigmay,angle'''
    return amplitude*(np.exp(-(x-xorigin)**2/(2.*sigma**2))*np.exp(-(y-yorigin)**2/(2.*sigma**2)))
def gaussian_1var(p, x): #INPUT_MAT,xorigin,yorigin,sigma):
    '''xorigin,yorigin,sigmax,sigmay,angle'''
    xorigin,yorigin,sigma,amplitude= p
    x0=x[0]
    x1=x[1]
    x0=x0[:,None]
    x1=x1[None,:]
    return amplitude*(np.exp(-(x0-xorigin)**2/(2.*sigma**2))*np.exp(-(x1-yorigin)**2/(2.*sigma**2)))
def err(p, y, x):
    ''' 
    p is a tuple contatining the initial parameters.  p=(xorigin,yorigin,sigma, amplitude)
    y is the data we are fitting to (the dependent variable)
    x is the independent variable
    '''
    remander=y - gaussian_1var(p, x)
    return remander.ravel()