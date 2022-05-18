# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Software de Observaciones Sintéticas S.O.S.
# Line fitting functions
#
# Marcial Becerril, @ 19 January 2022
# Latest Revision: 19 Jan 2022, 19:12 GMT
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import numpy as np

from matplotlib import colors
from matplotlib.pyplot import *
from scipy.optimize import curve_fit

from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtCore import Qt, QObject, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon

import sos
from .misc.line_functions import *
from .misc.print_msg import *
from .misc.table_model import *

from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import simps

from datetime import datetime

from matplotlib.backends.backend_qt4agg import(
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)



class BaselineSpecWindow(QWidget):
    """
        Substract baselines
        Parameters
        ----------
            x : array
            y : array
            method : array
                Method of baseline substraction:
                1. line. Get the best fit line that fit several points
                2. polynomial N-degree. Best fit with a polynomial function of N-degree
                3. bls. BLS Algorithm
        ----------
    """
    # Signal to update data
    signal_baseline = QtCore.pyqtSignal(str)

    def __init__(self):

        super(BaselineSpecWindow, self).__init__()

        uic.loadUi("./sos/res/gui/baseline.ui", self)

        # Initialisation of variables
        self.inter = False
        self.selPointsFlag = False
        self.selRegionFlag = False
        self.method = None
        self.flag_apply = False
        self.iter_points = []

        # Assign buttons
        self.cancelButton.mousePressEvent = self.close_widget
        # Activate interaction
        self.interactButton.mousePressEvent = self.activate_interactive
        # Type of selection
        self.choosePointsButton.mousePressEvent = self.act_points_selection
        self.removePointsButton.mousePressEvent = self.act_region_selection
        # Selection method
        self.linealButton.mousePressEvent = self.linear_selection
        self.polyButton.mousePressEvent = self.poly_selection
        self.blsButton.mousePressEvent = self.bls_selection
        # Clear button
        self.clearButton.mousePressEvent = self.reset_canvas
        # Apply button
        self.applyButton.mousePressEvent = self.apply_baseline_correction
        # Accept button
        self.acceptButton.mousePressEvent = self.accept_baseline_correction

        #cid = self.f1.figure.canvas.mpl_connect('button_press_event', self.resDraw)


    def load_init_params(self, fig, ax, x, y, name, save):
        # Load initial params
        self.x = x
        self.y = y

        # File name
        self.nameLabel.setText(name)

        # Initialise corrected data array
        self.data_corrected = self.y.copy()

        # Get figure
        self.fig = fig
        self.fig.subplots_adjust(left=0.12, bottom=0.12, right=0.98,
                                top=0.98, wspace=None, hspace=None)
        self.ax = ax 
        
        # Save figure?
        self.save = save

        # Update plot
        self._addmpl(self.fig)

        # Initial plot
        self.initial_plot()


    def close_widget(self, event):
        # Disable graphs
        self.close()


    def activate_interactive(self, event):
        # Interactive activation
        self.inter = not self.inter
        if self.inter:
            if self.selPointsFlag or self.selRegionFlag:
                self._onclick_xy = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
            else:
                msg('Choose one selection mode', 'warn')
                self.inter = not self.inter
                return

            icon_img = './sos/res/icons/int_sel.png'
        else:
            if self.selPointsFlag or self.selRegionFlag:
                self.fig.canvas.mpl_disconnect(self._onclick_xy)

            icon_img = './sos/res/icons/int.png'

        self.interactButton.setIcon(QIcon(icon_img))


    def act_points_selection(self, event):
        # Points selection activated
        self.selection_settings(True)
        self._update_selected_plot(self.iter_points)


    def act_region_selection(self, event):
        # Region selection activated
        self.selection_settings(False)
        self._update_selected_plot(self.iter_points)


    def linear_selection(self, event):
        # Linear baseline method
        self.baseline_method('linear')


    def poly_selection(self, event):
        # Linear baseline method
        self.baseline_method('poly')


    def bls_selection(self, event):
        # Linear baseline method
        self.baseline_method('bls')


    def baseline_method(self, method):

        self.method = method

        if self.method == 'linear':
            linear = './sos/res/icons/lineal_icon_sel.png'
            poly = './sos/res/icons/poly_curve.png' 
            bls = './sos/res/icons/bls_icon.png'    
            # Disable the other functions
            self.nDegreeBox.setEnabled(False)
            self.lamdbaBLSEdit.setEnabled(False)
            self.pBLSEdit.setEnabled(False)
            self.iterBLSEdit.setEnabled(False)

        elif self.method == 'poly':
            linear = './sos/res/icons/lineal_icon.png'
            poly = './sos/res/icons/poly_curve_sel.png' 
            bls = './sos/res/icons/bls_icon.png' 
            # Disable the other functions
            self.nDegreeBox.setEnabled(True)
            self.lamdbaBLSEdit.setEnabled(False)
            self.pBLSEdit.setEnabled(False)
            self.iterBLSEdit.setEnabled(False)

        elif self.method == 'bls':
            linear = './sos/res/icons/lineal_icon.png'
            poly = './sos/res/icons/poly_curve.png' 
            bls = './sos/res/icons/bls_icon_sel.png' 
            # Disable the other functions
            self.nDegreeBox.setEnabled(False)
            self.lamdbaBLSEdit.setEnabled(True)
            self.pBLSEdit.setEnabled(True)
            self.iterBLSEdit.setEnabled(True)

        self.linealButton.setIcon(QIcon(linear))
        self.polyButton.setIcon(QIcon(poly))
        self.blsButton.setIcon(QIcon(bls))


    def selection_settings(self, ptsFlag):
        # Grpah Selection Configuration
        self.selPointsFlag = ptsFlag
        self.selRegionFlag = not self.selPointsFlag

        if self.selPointsFlag:
            points = './sos/res/icons/choosePoints_sel.png'
            region = './sos/res/icons/removePoints.png'
        else:
            points = './sos/res/icons/choosePoints.png'
            region = './sos/res/icons/removePoints_sel.png'

        self.choosePointsButton.setIcon(QIcon(points))
        self.removePointsButton.setIcon(QIcon(region))


    def apply_baseline_correction(self, event):
        # Baseline correction
        if self.method == 'bls':
            l = float(self.lamdbaBLSEdit.text())
            p = float(self.pBLSEdit.text())
            n = int(float((self.iterBLSEdit.text())))
            self.iterBLSEdit.setText(str(n))
            y_baseline = baseline_als_optimized(self.y, l, p, niter=n)
        
        else:
            points = self.iter_points
            points = np.sort(points)

            if self.selPointsFlag:
                x_filtered = []
                y_filtered = []
                # Extracting data [points]
                for i in range(len(points)):
                    x_filtered.append(self.x[points[i]])
                    y_filtered.append(self.y[points[i]])

            elif self.selRegionFlag:
                # Extracting data [region]
                adquire_data = False
                x_mask = [True]*len(self.x)
                y_mask = [True]*len(self.y)
                for i in range(len(points)):
                    if adquire_data:
                        x_mask[points[i-1]:points[i]] = [False]*(points[i]-points[i-1])
                        y_mask[points[i-1]:points[i]] = [False]*(points[i]-points[i-1]) 
                    adquire_data = not adquire_data

                x_filtered = np.array(self.x)[x_mask]
                y_filtered = np.array(self.y)[y_mask]

            else:
                msg('Choose one selection mode', 'warn')
                return

            x_baseline = self.x

            if self.method == 'linear':
                y_baseline = poly_baseline(x_filtered, y_filtered, 1, x_baseline)

            elif self.method == 'poly':
                degree = self.nDegreeBox.value()
                y_baseline = poly_baseline(x_filtered, y_filtered, degree, x_baseline)
            
        # Set flag
        self.flag_apply = True

        data_corrected = self.y - y_baseline

        # Update data with baseline substrated
        self.data_corrected = data_corrected

        self._update_plot(y_baseline, data_corrected)


    def accept_baseline_correction(self, event):
        # Applying baseline correction
        if not self.flag_apply:
            self.apply_baseline_correction(event)
        
        #self.signal_baseline.emit(self.kind)

        if self.save:
            now = datetime.now()
            name = now.strftime("%d-%m-%Y_%H-%M-%S")
            self.fig.savefig('fig_'+name+'_bl.png')

        self.close()


    def _onclick(self, event):
        """
            On click event to select lines
        """
        if event.inaxes == self.ax:
            # Left-click
            if event.button == 1:
                ix, iy = event.xdata, event.ydata
                # Add peaks
                xarray = np.where(self.x>ix)[0]
                if len(xarray) > 0:
                    xpos = xarray[0]
                else:
                    xpos = len(self.x)-1
                self.iter_points.append(xpos)

                self.flag_apply = False

            # Right-click
            elif event.button == 3:
                ix, iy = event.xdata, event.ydata
                popt = []
                # Remove points
                # Define threshold
                thresh = 5*np.mean(np.diff(self.x))
                xlines = np.where((ix >= (np.array(self.x)[self.iter_points] - thresh)) & 
                               (ix < (np.array(self.x)[self.iter_points] + thresh)))[0]

                try:
                    if len(xlines) > 0:
                        x_min = np.argmin(np.abs((np.array(self.x)[np.array(self.iter_points)[xlines]] - ix)))
                        if self.selRegionFlag:
                            self.iter_points.remove(self.iter_points[xlines[x_min]])
                        else:
                            ylines = np.where((iy >= (np.array(self.y)[self.iter_points] - thresh)) & 
                                              (iy < (np.array(self.y)[self.iter_points] + thresh)))

                            if len(ylines) > 0:
                                y_min = np.argmin(np.abs((np.array(self.y)[np.array(self.iter_points)[ylines]] - iy)))
                                self.iter_points.remove(self.iter_points[xlines[y_min]])
                        
                        self.flag_apply = False
                except:
                    pass

            # Update plot
            self._update_selected_plot(self.iter_points)


    def _update_selected_plot(self, points):
        """
            Update selected Points/Region in the canvas
        """
        self.ax.clear()

        # Label axes
        ux_label = self.ux
        if self.ux:
            ux_label = '['+ux_label+']'
        uy_label = self.uy
        if self.uy:
            uy_label = '['+uy_label+']'

        self.ax.set_xlabel(r''+ux_label)
        self.ax.set_ylabel(r'Temperature '+uy_label)

        self.ax.plot(self.x, self.y, 'k')
        for i in range(len(points)):
            if self.selPointsFlag:
                self.ax.plot(self.x[points[i]], self.y[points[i]], 'r+')
            elif self.selRegionFlag:
                self.ax.axvline(self.x[points[i]], color='r', linewidth=1)

        self.ax.grid()

        self.fig.canvas.draw_idle()


    def _update_plot(self, baseline, data_corrected):
        """
            Update baseline in the canvas
        """
        self.ax.clear()

        # Label axes
        ux_label = self.ux
        if self.ux:
            ux_label = '['+ux_label+']'
        uy_label = self.uy
        if self.uy:
            uy_label = '['+uy_label+']'

        self.ax.set_xlabel(r''+ux_label)
        self.ax.set_ylabel(r'Temperature '+uy_label)

        self.ax.plot(self.x, self.y, 'k', linewidth=0.75)
        self.ax.plot(self.x, baseline, 'c-.', linewidth=0.75)
        self.ax.plot(self.x, data_corrected, 'r')
        self.ax.grid()

        self.fig.canvas.draw_idle()


    def initial_plot(self):
        """
            Initial plot
        """
        self.ax.clear()
        self.ax.plot(self.x, self.y, 'k')

        # Label axes
        ux_label = self.ux
        if self.ux:
            ux_label = '['+ux_label+']'
        uy_label = self.uy
        if self.uy:
            uy_label = '['+uy_label+']'

        self.ax.set_xlabel(r''+ux_label)
        self.ax.set_ylabel(r'Temperature '+uy_label)

        self.ax.grid()

        self.fig.canvas.draw_idle()


    def reset_canvas(self, event):
        # Restart to initial plot
        self.initial_plot()

        self.iter_points = []


    def _addmpl(self, fig):
        
        self.canvas = FigureCanvas(fig)
        self.plotLayout.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas,
           self, coordinates=True)
        self.plotLayout.addWidget(self.toolbar)


    def _rmmpl(self):
        self.plotLayout.removeWidget(self.canvas)
        self.canvas.close()
        self.plotLayout.removeWidget(self.toolbar)
        self.toolbar.close()