from process.BaseProcess import BaseProcess
import pyqtgraph as pg
from PyQt4.QtGui import *
import global_vars as g
from leastsqbound import leastsqbound
import numpy as np
from scipy.ndimage.interpolation import shift
from window import Window

class Drift_Correction(BaseProcess):
    '''drift_correction():
-In Progress
    '''
    def __init__(self):
        super().__init__()
        findButton = QPushButton("Find Centers")
        findButton.pressed.connect(self.call)
        alignButton = QPushButton("Correct Drift")
        alignButton.pressed.connect(self.correct)
        self.items.append({'name':'findButton','string':'Locate centroids','object':findButton})
        self.items.append({'name':'alignButton','string':'Correct For Drift','object':alignButton})

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

    def correct(self):
        diffs = np.zeros((len(self.rois), np.size(self.rois[0]['centers'], 0), 2))
        
        for i, roi in enumerate(self.rois):
            centers = np.array(roi['centers'])
            diffs[i] = -1 * np.round(centers - centers[0])
        diffs = np.mean(diffs, 0)
        im = g.m.currentWindow.image
        for i, sh in enumerate(diffs):
            g.m.statusBar().showMessage("shifting frame %d of %d" % (i, len(diffs)))
            im[i] = shift(im[i], sh)
            QApplication.instance().processEvents()
        return Window(im)


    def plotPoints(self, i):
        if self.scatter not in g.m.currentWindow.imageview.view.addedItems:
            g.m.currentWindow.imageview.addItem(self.scatter)
        self.scatter.setData(pos=[r['centers'][i] for r in self.rois], brush=[r['roi'].color for r in self.rois], size=1)

    def call(self):
        self.find_centers()
        g.m.currentWindow.sigTimeChanged.connect(self.plotPoints)

        self.p1.clear()
        self.p2.clear()
        for r in self.rois:
            centers = np.array(r['centers'])
            self.p1.plot(y=centers[:, 0] / np.average(centers[:, 0]), pen=QPen(r['roi'].color))
            self.p2.plot(y=centers[:, 1] / np.average(centers[:, 1]), pen=QPen(r['roi'].color))

    def find_centers(self):
        win = g.m.currentWindow
        im = win.image
        mx,my=im[0,:,:].shape
        self.rois = []
        for roi in g.m.currentWindow.rois:
            mask = roi.mask + roi.minn
            mask=mask[(mask[:,0]>=0)*(mask[:,0]<mx)*(mask[:,1]>=0)*(mask[:,1]<my)]

            xx=mask[:,0]; yy=mask[:,1]

            centers = []
            I = im[0, xx, yy]
            sh =  (max(xx) - min(xx) + 1, max(yy) - min(yy) + 1)
            I = np.reshape(I,sh)
            init_x, init_y = np.where(I==np.max(I))
            init_x = init_x[0]
            init_y = init_y[0]
            for frame in range(len(im)):
                I = im[frame, xx, yy]
                sh =  (max(xx) - min(xx) + 1, max(yy) - min(yy) + 1)
                I = np.reshape(I,sh)
                vals = fitGaussian(I, [init_x, init_y, 1, 1])
                x0, y0, a, b = vals[0]
                init_x, init_y = [x0, y0]
                centers.append([x0 + roi.minn[0], y0 + roi.minn[1]])
            self.rois.append({'roi': roi, 'centers': centers})

drift_correction = Drift_Correction()

def fitGaussian(I=None, p0=None, bounds=None, nGaussians=1):
    '''
    Takes an nxm matrix and returns an nxm matrix which is the gaussian fit
    of the first.  p0 is a list of parameters [xorigin, yorigin, sigma,amplitude]
    0-19 should be [-.2889 -.3265 -.3679 -.4263 -.5016 -.6006 ... -.0228 .01913]
    '''

    x=np.arange(I.shape[0])
    y=np.arange(I.shape[1])
    X=[x,y]
    p0=[round(p,3) for p in p0] 
    if nGaussians==1:
        p, cov_x, infodic, mesg, ier = leastsqbound(err, p0,args=(I,X),bounds = bounds,ftol=.0000001,full_output=True)
        #xorigin,yorigin,sigmax,sigmay,angle,amplitude=p
        I_fit=gaussian(x[:,None], y[None,:],*p)
        return p, I_fit, I_fit
    elif nGaussians>=2:
        p, cov_x, infodic, mesg, ier = leastsqbound(err2, p0,args=(I,X),bounds = bounds,ftol=.0000001,full_output=True)
        #xorigin,yorigin,sigmax,sigmay,angle,amplitude=p
        I_fit=gaussian(x[:,None], y[None,:],*(p[:4]))
        I_fit2=gaussian2(x[:,None], y[None,:],*p)
        return p[:4], I_fit, I_fit2
        
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