__author__ = 'Maxime'

# -*- coding: utf-8 -*-

from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import os
import sys
import matplotlib
import numpy as np
from scipy import sparse
import pickle
import time
import copy
import struct
import csv

matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from scipy import signal
import scipy.io
import pyedflib
import datetime
from Graphical.EEGViewer import lfpViewer_EEG
from Graphical.Modify_Each_NMM_param import Modify_1_NMM
# from Modify_X_NMM_at_once import Modify_X_NMM
from Graphical.Modify_X_NMM_VTK import Modify_X_NMM
from numba import guvectorize, njit
from Graphical.Graph_viewer3D_VTK5 import Graph_viewer3D_VTK
from Graphical.Graph_EField_VTK import Graph_EField_VTK
from scipy.optimize import curve_fit
import platform
from Computation import Connectivity, Electrode, RecordedPotential
from Tissue import CreateColumn


def iir_band_filter(ite_data,fs,order,lowcut,highcut,zerophase,btype,ftype,rps=None):
        fe = fs/2.0
        low = lowcut/fe
        high = highcut/fe
        if low<0:
            low=0
        if high>1:
            high=1
        if ftype == "cheby1":
            rp = rps
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = rps
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = rps[0]
            rs = rps[1]
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk")
        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data

def signalfilterbandpass(LFPs,Fs,lowcut, highcut, order):

    if len(LFPs.shape) ==1:
        LFPs = LFPs[np.newaxis, :]

    if order >10:
        order=10
    if order <0:
        order=0
    if lowcut <=0:
        lowcut=1/Fs
    if highcut > Fs/2:
        highcut = Fs/2-1
    for idx_lfp, lfp in enumerate(LFPs):

        # LFPs[idx_lfp]=butter_bandpass_filter(lfp, lowcut, highcut, Fs, order)

        # h = filtreRIF("PBande",lowcut / Fs , highcut/ Fs , order ,"hann")
        # LFPs[idx_lfp] = scipy.signal.convolve(lfp,h,mode='same')

        LFPs[idx_lfp]=iir_band_filter(lfp,Fs,order,lowcut,highcut,0,'bandpass','bessel',rps=None)
    return LFPs

class Spoiler(QWidget):
    def __init__(self, parent=None, title='', animationDuration=300):
        """
        References:
            # Adapted from c++ version
            http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt
        """
        super(Spoiler, self).__init__(parent=parent)

        self.animationDuration = animationDuration
        self.toggleAnimation = QParallelAnimationGroup()
        self.contentArea = QScrollArea()
        self.headerLine = QFrame()
        self.toggleButton = QToolButton()
        self.mainLayout = QGridLayout()

        toggleButton = self.toggleButton
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(Qt.ArrowType.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)

        headerLine = self.headerLine
        headerLine.setFrameShape(QFrame.Shape.HLine)
        headerLine.setFrameShadow(QFrame.Shadow.Sunken)
        headerLine.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        self.contentArea.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # start out collapsed
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        # let the entire widget grow and shrink with its content
        toggleAnimation = self.toggleAnimation
        toggleAnimation.addAnimation(QPropertyAnimation(self, b"minimumHeight"))
        toggleAnimation.addAnimation(QPropertyAnimation(self, b"maximumHeight"))
        toggleAnimation.addAnimation(QPropertyAnimation(self.contentArea, b"maximumHeight"))
        # don't waste space
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, Qt.AlignmentFlag.AlignLeft)
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        row += 1
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def start_animation(checked):
            arrow_type = Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
            direction = QAbstractAnimation.Direction.Forward if checked else QAbstractAnimation.Direction.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()

        self.toggleButton.clicked.connect(start_animation)
        toggleButton.setChecked(True)
        start_animation(True)

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentArea.destroy()
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()
        for i in range(self.toggleAnimation.animationCount() - 1):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)

if platform.system() == "Windows":
    pass
elif platform.system() == "Darwin":
    matplotlib.rcParams.update({'font.size': 6})
elif platform.system() == "Linux":
    pass


def peakdet(v, delta, x=None):
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)




@guvectorize(["float64[:,:],int64, int64, float64, float64, int64, float64, int64, int64, float64, float64[:,:]"],
             '(n,m),(),(),(),(),(),(),(),(),()->(n,m)')
def Generate_Stim_Signals(Stim_Signals_in, seed, nbOfSamplesStim, i_inj, tau, nbStim, varianceStim, nb_Stim_Signals,
                          nbEch, dt, Stim_Signals_out):
    n = int(1. * nbEch / 2.)
    t = np.arange(nbEch) * dt

    if not seed == 0:
        np.random.seed(seed)
    # else:
    #     np.random.seed()
    for St in range(nb_Stim_Signals):
        y2 = np.zeros(nbEch)
        for i in range(nbStim):

            y = np.zeros(t.shape)

            intervalle_dt = int(np.round((np.random.normal(0, varianceStim))))

            if ((n + intervalle_dt) < 0):
                intervalle_dt = -n
            if ((n + intervalle_dt + nbOfSamplesStim) > nbEch):
                intervalle_dt = nbEch - n - nbOfSamplesStim - 1

            for tt, tp in enumerate(t):
                if (tt > (n + intervalle_dt)):
                    y[tt] = (1. - np.exp(-(tp - (n + intervalle_dt) * dt) / tau)) * i_inj
                if (tt > (n + intervalle_dt + nbOfSamplesStim)):
                    y[tt] = (np.exp(-(tp - ((n + intervalle_dt) + nbOfSamplesStim) * dt) / tau)) * y[
                        n + intervalle_dt + nbOfSamplesStim - 1]

                y2[tt] += y[tt]

        Stim_Signals_out[St, :] = y2
    # return Stim_Signals


@guvectorize(["int64[:], int64, int64, int64, int64[:]"], '(n),(),(),()->(n)')
def pickcell_simple_fun(size, loop, low, high, pickedcell):
    while loop > 0:
        # cell = np.random.randint(low, high, size=1)
        cell = np.random.randint(low, high)
        if cell in pickedcell:
            pass
        else:
            pickedcell[loop - 1] = cell
            loop -= 1


@guvectorize(["int64[:], int64, float64, float64, float64[:], int64[:], int64[:]"], '(n),(),(),(),(m),(k)->(n)')
def pickcell_gauss_fun(size, loop, mean, var, CellDistances, sortedIndex, pickedcell):
    while loop > 0:
        l = abs(np.random.normal(mean, var))
        cellInd = np.where(CellDistances[sortedIndex] >= l)[0]
        if cellInd.size == 0:
            continue
        else:
            cellInd = cellInd[0]
        cell = sortedIndex[cellInd]
        if not cell in pickedcell:
            pickedcell[loop - 1] = cell
            sortedIndex = np.delete(sortedIndex, cellInd)
            loop -= 1


@guvectorize(["float64[:], int64, int64[:]"], '(n),()->(n)')
def argsort_list_fun(CellDistances, plus, sortedIndex):
    sortedIndex = np.argsort(CellDistances) + plus


class LineEdit(QLineEdit):
    KEY = Qt.Key.Key_Return

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegularExpressionValidator(QRegularExpression("[+-]?\\d*[\\.]?\\d+"))
        QREV.setLocale(QLocale(QLocale.Language.English))
        self.setValidator(QREV)


class LineEdit_Int(QLineEdit):
    KEY = Qt.Key.Key_Return

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegularExpressionValidator(QRegularExpression("[+-]?\\d+"))
        QREV.setLocale(QLocale(QLocale.Language.English))
        self.setValidator(QREV)


def set_QPushButton_background_color(button=None, color=None):
    if color == None or button == None:
        return
    else:
        button.setAutoFillBackground(True)
        values = "{r}, {g}, {b} ".format(r=color.red(),
                                         g=color.green(),
                                         b=color.blue())
        button.setStyleSheet("QPushButton { background-color: rgb(" + values + "); }")


def label_color_clicked(event, button):
    color = QColor(button.palette().button().color())
    colordial = QColorDialog(color)
    colordial.exec()
    selectedcolor = colordial.currentColor()
    colordial.close()
    set_QPushButton_background_color(button, selectedcolor)
    pass


def Layout_grid_Label_Edit(label=['None'], edit=['None']):
    widget = QWidget()
    layout_range = QVBoxLayout()
    # layout_range.setContentsMargins(5,5,5,5)

    grid = QGridLayout()
    grid.setContentsMargins(5, 5, 5, 5)
    widget.setLayout(grid)
    layout_range.addLayout(grid)
    Edit_List = []
    for idx in range(len(label)):
        Label = QLabel(label[idx])
        Edit = LineEdit(edit[idx])
        grid.addWidget(Label, idx, 0)
        grid.addWidget(Edit, idx, 1)
        Edit_List.append(Edit)
    return widget, Edit_List


class Colonne_cortical_Thread():
    finished = pyqtSignal()
    updateTime = pyqtSignal(float)

    def __init__(self, CortexClass, type=1):
        # QThread.__init__(self)
        self.CC = CortexClass(type)

        self.percent = 0.
        self.T = 0
        self.dt = 0
        self.Stim_Signals = []

    @pyqtSlot(float)
    def updatePercent(self, pourcent):
        self.percent = pourcent
        self.updateTime.emit(self.percent)

    # def __del__(self):
    #     self.wait()

    def get_percentage(self):
        return self.percent

    def arret(self):
        self.C.Stop = False


class ModelMicro_GUI(QMainWindow):
    def __init__(self, parent=None):
        super(ModelMicro_GUI, self).__init__()
        self.parent = parent

        if getattr(sys, 'frozen', False):
            self.application_path = os.path.dirname(sys.executable)
        elif __file__:
            self.application_path = os.path.dirname(__file__)
        sys.path.append(self.application_path)
        self.setWindowIcon(QIcon('NeoCoMM_Logo2.png'))
        from Tissue.CorticalColumn import CorticalColumn
        self.Colonne = CorticalColumn
        self.Colonne_cortical_Thread = Colonne_cortical_Thread(self.Colonne,type=1)
        self.CC = self.Colonne_cortical_Thread.CC
        self.CC.updateTime.something_happened.connect(self.updateTime)

        self.dt = 1 / 25
        self.T = 200
        self.nbEch = int(self.T / self.dt)
        self.List_PYR_Stimulated = []

        self.x = 0.
        self.y = 0.
        self.z = 0.

        self.electrode_pos = [0., 0., 2082.]

        self.createCells = True

        # some variable for widget's length
        self.Qbox_H = 30
        self.Qbox_W = 60
        self.Qedit_H = 30
        self.Qedit_W = 60

        # Menu
        # set actions
        extractLoad_Model = QAction("Load Model", self)
        extractLoad_Model.triggered.connect(self.LoadModel)

        extractLoad_Simul = QAction("Load Simulation", self)
        extractSave_Simul = QAction("Save Simulation", self)
        extractLoad_Simul.triggered.connect(self.LoadSimul)
        extractSave_Simul.triggered.connect(self.SaveSimul)

        extractLoad_Res = QAction("Load Results", self)
        extractSave_Res = QAction("Save Results", self)
        # extractLoad_Res.triggered.connect(self.LoadRes)
        extractSave_Res.triggered.connect(self.SaveRes)

        menubar = self.menuBar()
        # menubar.setNativeMenuBar(True)
        fileLoad = menubar.addMenu('&Load/Save')
        fileLoad.addAction(extractLoad_Model)
        fileLoad.addSeparator()
        fileLoad.addAction(extractLoad_Simul)
        fileLoad.addAction(extractSave_Simul)
        fileLoad.addSeparator()
        # fileLoad.addAction(extractLoad_Res)
        fileLoad.addAction(extractSave_Res)


        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        #### global layout
        self.mainHBOX = QHBoxLayout()
        self.setmarginandspacing(self.mainHBOX)
        ###

        wid_Param_VBOX = QWidget()
        self.Param_VBOX = QVBoxLayout()
        self.Param_VBOX.setAlignment(Qt.AlignmentFlag.AlignTop)
        wid_Param_VBOX.setLayout(self.Param_VBOX)
        # wid_Param_VBOX.setMaximumWidth(300)
        self.setmarginandspacing(self.Param_VBOX)

        # call column layout
        self.set_Param_VBOX()

        self.Vsplitter_middle = QSplitter(Qt.Orientation.Vertical)
        self.mascene_EEGViewer = lfpViewer_EEG()
        self.mascene_LFPViewer = LFPViewer(self)
        self.mascene_LFPViewer.clearfig()
        self.Vsplitter_middle.addWidget(self.mascene_EEGViewer)
        self.Vsplitter_middle.addWidget(self.mascene_LFPViewer)
        self.Vsplitter_middle.setStretchFactor(4, 0)
        self.Vsplitter_middle.setStretchFactor(1, 0)
        self.Vsplitter_middle.setSizes([1500, int(1500 / 4)])

        self.Vsplitter = QSplitter(Qt.Orientation.Vertical)

        self.GraphWidget = QWidget()
        self.GraphLayout = QVBoxLayout()
        self.Graph_viewer = Graph_viewer3D_VTK(self)
        self.GraphinfoLayout = QHBoxLayout()
        labelr, r_e = Layout_grid_Label_Edit(label=['r'], edit=['50'])
        labell, l_e = Layout_grid_Label_Edit(label=['line'], edit=['5'])
        labels, s_e = Layout_grid_Label_Edit(label=['scale'], edit=['50'])
        self.r_e = r_e[0]
        self.l_e = l_e[0]
        self.s_e = s_e[0]
        self.RedrawVTK_PB = QPushButton('Redraw')
        self.GraphinfoLayout.addWidget(labelr)
        self.GraphinfoLayout.addWidget(self.r_e)
        self.GraphinfoLayout.addWidget(labell)
        self.GraphinfoLayout.addWidget(self.l_e)
        self.GraphinfoLayout.addWidget(labels)
        self.GraphinfoLayout.addWidget(self.s_e)
        self.GraphinfoLayout.addWidget(self.RedrawVTK_PB)
        self.GraphLayout.addLayout(self.GraphinfoLayout)
        self.GraphLayout.addWidget(self.Graph_viewer)
        self.GraphWidget.setLayout(self.GraphLayout)
        self.r_e.editingFinished.connect(self.updateVTKinfo)
        self.l_e.editingFinished.connect(self.updateVTKinfo)
        self.s_e.editingFinished.connect(self.updateVTKinfo)
        self.RedrawVTK_PB.clicked.connect(self.Graph_viewer.draw_Graph)

        self.masceneCM = CMViewer(self)
        self.VStimWidget = QWidget()
        self.VStimLayout = QVBoxLayout()
        self.masceneStim = StimViewer(self)
        self.DisplayStim_PB = QPushButton('DisplayStim')
        self.DisplayStim_PB.clicked.connect(self.DisplayStim_func)
        self.VStimLayout.addWidget(self.DisplayStim_PB)
        self.VStimLayout.addWidget(self.masceneStim)
        self.VStimWidget.setLayout(self.VStimLayout)
        self.Vsplitter.addWidget(self.GraphWidget)
        self.Vsplitter.addWidget(self.masceneCM)
        self.Vsplitter.addWidget(self.VStimWidget)
        self.Vsplitter.setStretchFactor(0, 0)
        self.Vsplitter.setStretchFactor(1, 0)
        self.Vsplitter.setStretchFactor(2, 0)
        self.Vsplitter.setStretchFactor(3, 0)
        self.Vsplitter.setSizes([1500, 1500, 1500, 1500])

        ### add  vectical global splitters
        self.mainsplitter = QSplitter(Qt.Orientation.Horizontal)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        widget = QWidget()
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(wid_Param_VBOX)
        wid_Param_VBOX.setFixedWidth(400)
        scroll.setWidget(widget)
        # scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400 + 24)
        scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidgetResizable(True)
        self.mainsplitter.addWidget(scroll)

        self.mainsplitter.addWidget(self.Vsplitter_middle)
        self.mainsplitter.addWidget(self.Vsplitter)
        self.mainsplitter.setStretchFactor(0, 1)
        self.mainsplitter.setStretchFactor(1, 3)
        self.mainsplitter.setStretchFactor(2, 1)
        self.mainsplitter.setSizes([400, 1500 * 3, 1500])

        self.mainHBOX.addWidget(self.mainsplitter)

        self.centralWidget.setLayout(self.mainHBOX)





        self.NewModify1NMM = None
        self.NewModifyXNMM = None
        self.selected_cells = []
        # self.update_cellNnumber()

    def updateVTKinfo(self):
        self.Graph_viewer.LinewidthfromGUI = float(self.l_e.text())
        self.Graph_viewer.radiuswidthfromGUI = float(self.r_e.text())
        self.Graph_viewer.setScales(float(self.s_e.text()))

    @pyqtSlot(float)
    def updateTime(self, pourcent):
        ts = time.time() - self.t0
        tf = ts / (pourcent + 0.00000001)
        tr = tf - ts
        self.msg.setText("Computation in progress\nPlease wait.\nTime spend : " + str(
            datetime.timedelta(seconds=int(ts))) + "\nTime remaining : " + str(datetime.timedelta(seconds=int(tr))))
        self.parent.processEvents()

    def DisplayStim_func(self):
        # self.Generate_Stim_Signals()
        self.masceneStim.update(self.CC.Stim_Signals, self.CC.Stim_InputSignals)

    def set_Param_VBOX(self):
        # type of tissue human rat mouse
        self.type_tissu_SP = Spoiler(title=r'type of tissue: human, rat or mouse')
        layout_type_tissu = QHBoxLayout()
        self.type_tissu_BG = QButtonGroup()
        self.type_tissu_Human = QRadioButton('Human')
        self.type_tissu_Human.setChecked(False)
        self.type_tissu_Rat = QRadioButton('Rat')
        self.type_tissu_Rat.setChecked(True)
        self.type_tissu_Mouse = QRadioButton('Mouse')
        self.type_tissu_Mouse.setChecked(False)
        self.type_tissu_BG.addButton(self.type_tissu_Human)
        self.type_tissu_BG.addButton(self.type_tissu_Rat)
        self.type_tissu_BG.addButton(self.type_tissu_Mouse)
        layout_type_tissu.addWidget(self.type_tissu_Human)
        layout_type_tissu.addWidget(self.type_tissu_Rat)
        layout_type_tissu.addWidget(self.type_tissu_Mouse)
        self.type_tissu_SP.setContentLayout(layout_type_tissu)
        self.type_tissu_SP.toggleButton.click()
        self.type_tissu_Human.toggled.connect(self.ChangeModel_type)
        self.type_tissu_Rat.toggled.connect(self.ChangeModel_type)
        self.type_tissu_Mouse.toggled.connect(self.ChangeModel_type)

        # tissue size
        self.tissue_size_GB = Spoiler(title=r'tissue information')
        labelD, D_e = Layout_grid_Label_Edit(label=[u'cylinder diameter / Square XY length / Rectangle X length (\u00B5m)'],
                                             edit=[str(self.CC.D)])
        labelL, L_e = Layout_grid_Label_Edit(label=[u'cylinder length / /Rectangle Y length (\u00B5m)'],
                                             edit=[str(self.CC.L)])
        labelCourbure, Courbure_e = Layout_grid_Label_Edit(label=['Distance for curvature'],
                                                           edit=[str(2000)])
        self.D_e = D_e[0]
        self.L_e = L_e[0]
        self.C_e = Courbure_e[0]

        self.Layer_d1_l = LineEdit_Int(str(self.CC.Layer_d[0]))
        self.Layer_d23_l = LineEdit_Int(str(self.CC.Layer_d[1]))
        self.Layer_d4_l = LineEdit_Int(str(self.CC.Layer_d[2]))
        self.Layer_d5_l = LineEdit_Int(str(self.CC.Layer_d[3]))
        self.Layer_d6_l = LineEdit_Int(str(self.CC.Layer_d[4]))

        self.Apply_tissue_PB = QPushButton('Apply Tissue update')
        # grid = QGridLayout()

        self.Density_e = LineEdit('3954334')
        self.nbcell_e = LineEdit_Int('1000')

        self.get_XYZ_PB = QPushButton('Get XYZ')
        self.get_Density_PB = QPushButton('Get Density')
        self.get_nbcell_PB = QPushButton('Get Nb cells')
        line = 0
        grid = QGridLayout()
        grid.addWidget(labelD, line, 0, 1, 4)
        grid.addWidget(self.D_e, line, 4)
        line += 1
        grid.addWidget(labelL, line, 0, 1, 4)
        grid.addWidget(self.L_e, line, 4)
        line += 1
        grid.addWidget(labelCourbure, line, 0, 1, 4)
        grid.addWidget(self.C_e, line, 4)

        line += 1
        grid.addWidget(QLabel('thick 1'), line, 0)
        grid.addWidget(QLabel('thick 2/3'), line, 1)
        grid.addWidget(QLabel('thick 4'), line, 2)
        grid.addWidget(QLabel('thick 5'), line, 3)
        grid.addWidget(QLabel('thick 6'), line, 4)

        line += 1
        grid.addWidget(self.Layer_d1_l, line, 0)
        grid.addWidget(self.Layer_d23_l, line, 1)
        grid.addWidget(self.Layer_d4_l, line, 2)
        grid.addWidget(self.Layer_d5_l, line, 3)
        grid.addWidget(self.Layer_d6_l, line, 4)

        line += 1
        grid.addWidget(self.Apply_tissue_PB, line, 1, 1, 3)

        self.Apply_tissue_PB.clicked.connect(self.set_tissue_func)

        self.tissue_size_GB.setContentLayout(grid)
        self.tissue_size_GB.toggleButton.click()

        # %tage
        self.pourcentageCell_GB = Spoiler(title=r'% de cell')
        labelnb1 = QLabel('nb cells 1')
        labelnb23 = QLabel('nb cells 2/3')
        labelnb4 = QLabel('nb cells 4')
        labelnb5 = QLabel('nb cells 5')
        labelnb6 = QLabel('nb cells 6')
        labelnbtotal = QLabel('nb total')
        self.nbcellsnb1 = LineEdit_Int(str(int(self.CC.Layer_nbCells[0])))
        self.nbcellsnb23 = LineEdit_Int(str(int(self.CC.Layer_nbCells[1])))
        self.nbcellsnb4 = LineEdit_Int(str(int(self.CC.Layer_nbCells[2])))
        self.nbcellsnb5 = LineEdit_Int(str(int(self.CC.Layer_nbCells[3])))
        self.nbcellsnb6 = LineEdit_Int(str(int(self.CC.Layer_nbCells[4])))
        self.nbcellsnbtotal = LineEdit_Int(str(int(np.sum(self.CC.Layer_nbCells))))

        self.nbcellsnb1.returnPressed.connect(lambda s='1': self.Nb_Cell_Changed(s))
        self.nbcellsnb23.returnPressed.connect(lambda s='23': self.Nb_Cell_Changed(s))
        self.nbcellsnb4.returnPressed.connect(lambda s='4': self.Nb_Cell_Changed(s))
        self.nbcellsnb5.returnPressed.connect(lambda s='5': self.Nb_Cell_Changed(s))
        self.nbcellsnb6.returnPressed.connect(lambda s='6': self.Nb_Cell_Changed(s))
        self.nbcellsnbtotal.returnPressed.connect(lambda s='total': self.Nb_Cell_Changed(s))

        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(labelnb1, 0, 0)
        grid.addWidget(labelnb23, 0, 1)
        grid.addWidget(labelnb4, 0, 2)
        grid.addWidget(labelnb5, 0, 3)
        grid.addWidget(labelnb6, 0, 4)
        grid.addWidget(labelnbtotal, 0, 5)
        grid.addWidget(self.nbcellsnb1, 1, 0)
        grid.addWidget(self.nbcellsnb23, 1, 1)
        grid.addWidget(self.nbcellsnb4, 1, 2)
        grid.addWidget(self.nbcellsnb5, 1, 3)
        grid.addWidget(self.nbcellsnb6, 1, 4)
        grid.addWidget(self.nbcellsnbtotal, 1, 5)

        # nbconnexion
        label_source = QLabel('Layer')
        label_PYR = QLabel('PYR')
        label_PV = QLabel('PV')
        label_SST = QLabel('SST')
        label_VIP = QLabel('VIP')
        label_RLN = QLabel('RLN')
        label_1 = QLabel('1')
        label_23 = QLabel('2/3')
        label_4 = QLabel('4')
        label_5 = QLabel('5')
        label_6 = QLabel('6')

        grid.addWidget(label_source, 4, 1, 1, 5, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(label_PYR, 6, 0)
        grid.addWidget(label_PV, 7, 0)
        grid.addWidget(label_SST, 8, 0)
        grid.addWidget(label_VIP, 9, 0)
        grid.addWidget(label_RLN, 10, 0)
        grid.addWidget(label_1, 5, 1)
        grid.addWidget(label_23, 5, 2)
        grid.addWidget(label_4, 5, 3)
        grid.addWidget(label_5, 5, 4)
        grid.addWidget(label_6, 5, 5)
        # self.C = Column_morphology.Column(0)
        self.List_PYRpercent = []
        for l in range(len(self.CC.C.PYRpercent)):
            edit = LineEdit(str(self.CC.C.PYRpercent[l]))
            self.List_PYRpercent.append(edit)
            grid.addWidget(edit, 6, l + 1)

        self.List_PVpercent = []
        for l in range(len(self.CC.C.PVpercent)):
            edit = LineEdit(str(self.CC.C.PVpercent[l]))
            self.List_PVpercent.append(edit)
            grid.addWidget(edit, 7, l + 1)

        self.List_SSTpercent = []
        for l in range(len(self.CC.C.SSTpercent)):
            edit = LineEdit(str(self.CC.C.SSTpercent[l]))
            self.List_SSTpercent.append(edit)
            grid.addWidget(edit, 8, l + 1)

        self.List_VIPpercent = []
        for l in range(len(self.CC.C.VIPpercent)):
            edit = LineEdit(str(self.CC.C.VIPpercent[l]))
            self.List_VIPpercent.append(edit)
            grid.addWidget(edit, 9, l + 1)

        self.List_RLNpercent = []
        for l in range(len(self.CC.C.RLNpercent)):
            edit = LineEdit(str(self.CC.C.RLNpercent[l]))
            self.List_RLNpercent.append(edit)
            grid.addWidget(edit, 10, l + 1)

        # Compute cell number
        self.Apply_percentage_PB = QPushButton('Apply Percentage')
        grid.addWidget(self.Apply_percentage_PB, 11, 1, 1, 3)
        self.Apply_percentage_PB.clicked.connect(self.update_cellNnumber)

        self.pourcentageCell_GB.setContentLayout(grid)
        self.pourcentageCell_GB.toggleButton.click()
        # lambda s and d
        self.Lambda_S_D_GB = Spoiler(title=r'Lambda')

        Lambda_l = QHBoxLayout()
        self.ld_group = QButtonGroup(self)
        self.ld0 = QRadioButton("Lambda(f)")
        self.ld1 = QRadioButton("Constant")
        self.ld_group.addButton(self.ld0)
        self.ld_group.addButton(self.ld1)
        self.ld0.setChecked(True)
        Lambda_l.addWidget(self.ld0)
        Lambda_l.addWidget(self.ld1)



        self.grid = QGridLayout()
        # grid.addItem(Lambda_l)
        self.grid.setContentsMargins(5, 5, 5, 5)
        self.grid.addWidget(self.ld0,0,0,1,2)
        self.grid.addWidget(self.ld1,0,2,1,2)



        self.grid.addWidget(QLabel('Lambda S'), 1, 0)
        self.grid.addWidget(QLabel('II/III'), 2, 0)
        self.grid.addWidget(QLabel('IV'), 3, 0)
        self.grid.addWidget(QLabel('V'), 4, 0)
        self.grid.addWidget(QLabel('VI'), 5, 0)
        self.grid.addWidget(QLabel('TPC'), 1, 1)
        self.grid.addWidget(QLabel('UPC'), 1, 2)
        self.grid.addWidget(QLabel('IPC'), 1, 3)
        self.grid.addWidget(QLabel('BPC'), 1, 4)
        self.grid.addWidget(QLabel('SSC'),1, 5 )


        self.List_Lambda_S = []
        for l in range(4):
            for c in range(5):
                edit = LineEdit(str(0))
                self.List_Lambda_S.append(edit)
                self.grid.addWidget(edit, l+2, c + 1)

        for i in range(5):
            self.List_Lambda_S[i+0].setText(str(round(self.CC.List_Lambda_PC23_s[i][0],3)))
            self.List_Lambda_S[i+5].setText(str(round(self.CC.List_Lambda_PC4_s[i][0],3)))
            self.List_Lambda_S[i+10].setText(str(round(self.CC.List_Lambda_PC5_s[i][0],3)))
            self.List_Lambda_S[i+15].setText(str(round(self.CC.List_Lambda_PC6_s[i][0],3)))




        # self.grid.addWidget(QLabel(''), 5, 0)
        self.grid.addWidget(QLabel('Lambda D'), 6, 0)
        self.grid.addWidget(QLabel('II/III'), 7, 0)
        self.grid.addWidget(QLabel('IV'), 8, 0)
        self.grid.addWidget(QLabel('V'), 9, 0)
        self.grid.addWidget(QLabel('VI'), 10, 0)
        self.grid.addWidget(QLabel('TPC'), 6, 1)
        self.grid.addWidget(QLabel('UPC'), 6, 2)
        self.grid.addWidget(QLabel('IPC'), 6, 3)
        self.grid.addWidget(QLabel('BPC'), 6, 4)
        self.grid.addWidget(QLabel('SSC'), 6, 5 )


        self.List_Lambda_D = []
        for l in range(4):
            for c in range(5):
                edit = LineEdit(str(0))
                self.List_Lambda_D.append(edit)
                self.grid.addWidget(edit, l+7, c + 1)

        self.List_Lambda_D[i + 0].setText(str(round(self.CC.List_Lambda_PC23_d[i][0], 3)))
        self.List_Lambda_D[i + 5].setText(str(round(self.CC.List_Lambda_PC4_d[i][0], 3)))
        self.List_Lambda_D[i + 10].setText(str(round(self.CC.List_Lambda_PC5_d[i][0], 3)))
        self.List_Lambda_D[i + 15].setText(str(round(self.CC.List_Lambda_PC6_d[i][0], 3)))

        self.List_Lambda_IN = []
        for l in range(1,5):
            edit = LineEdit(str(round(self.CC.List_Lambda_PV[l][0],3)))
            self.List_Lambda_IN.append(edit)
            self.grid.addWidget(edit, l + 12,  1)
            edit = LineEdit(str(round(self.CC.List_Lambda_SST[l][0],3)))
            self.List_Lambda_IN.append(edit)
            self.grid.addWidget(edit, l + 12,  2)
            edit = LineEdit(str(round(self.CC.List_Lambda_VIP[l][0],3)))
            self.List_Lambda_IN.append(edit)
            self.grid.addWidget(edit, l + 12,  3)
            edit = LineEdit(str(round(self.CC.List_Lambda_RLN[l-1][0],3)))
            self.List_Lambda_IN.append(edit)
            self.grid.addWidget(edit, l + 12,  4)

        # self.grid.addWidget(QLabel(''), 5, 0)
        self.grid.addWidget(QLabel('Lambda IN'), 12, 0)
        self.grid.addWidget(QLabel('II/III'), 13, 0)
        self.grid.addWidget(QLabel('IV'), 14, 0)
        self.grid.addWidget(QLabel('V'), 15, 0)
        self.grid.addWidget(QLabel('VI'), 16, 0)
        self.grid.addWidget(QLabel('PV'), 12, 1)
        self.grid.addWidget(QLabel('SST'), 12, 2)
        self.grid.addWidget(QLabel('VIP'), 12, 3)
        self.grid.addWidget(QLabel('RLN'), 12, 4)


        # Compute cell number
        self.Apply_Lambda_PB = QPushButton('Apply Lambda')
        self.grid.addWidget(self.Apply_Lambda_PB, 17, 1, 1, 3)
        self.Apply_Lambda_PB.clicked.connect(self.update_lambda)

        self.Lambda_S_D_GB.setContentLayout(self.grid)
        self.Lambda_S_D_GB.toggleButton.click()


        # pyr percent type
        self.PCsubtypes_GB = Spoiler(title=r'PC subtypes')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        # grid.addWidget(QLabel('Lambda S'), 0, 0)
        grid.addWidget(QLabel('I'), 1, 0)
        grid.addWidget(QLabel('II/III'), 2, 0)
        grid.addWidget(QLabel('IV'), 3, 0)
        grid.addWidget(QLabel('V'), 4, 0)
        grid.addWidget(QLabel('VI'), 5, 0)
        grid.addWidget(QLabel('TPC'), 0, 1)
        grid.addWidget(QLabel('UPC'), 0, 2)
        grid.addWidget(QLabel('IPC'), 0, 3)
        grid.addWidget(QLabel('BPC'), 0, 4)
        grid.addWidget(QLabel('SSC'), 0, 5 )

        PCsubtypes_Percentage =self.CC.C.PCsubtypes_Percentage
        self.List_PCsubtypes = []
        for l in range(PCsubtypes_Percentage.shape[0]):
            for c in range(PCsubtypes_Percentage.shape[1]):
                edit = LineEdit(str(PCsubtypes_Percentage[l,c]))
                self.List_PCsubtypes.append(edit)
                grid.addWidget(edit, l+1, c + 1)

        # Compute cell number
        self.Apply_PCsubtypes_PB = QPushButton('Apply PC subtypes')
        grid.addWidget(self.Apply_PCsubtypes_PB, 6, 1, 1, 3)
        self.Apply_PCsubtypes_PB.clicked.connect(self.update_PCsubtypes)

        self.PCsubtypes_GB.setContentLayout(grid)
        self.ld0.toggled.connect(self.Change_Lambda_type)
        self.ld1.toggled.connect(self.Change_Lambda_type)
        self.PCsubtypes_GB.toggleButton.click()



        # Connection matrixcc
        self.Afferences_GB = Spoiler(title=r'afference matrix')
        Afferences_GB_l = QVBoxLayout()
        self.Afferences_PB = QPushButton('Get Afference matrix')
        self.Afference_group = QButtonGroup(self)
        self.r0 = QRadioButton("Use percentage of the total number of cell")
        self.r1 = QRadioButton("Use fixed number of connection")
        self.Afference_group.addButton(self.r0)
        self.Afference_group.addButton(self.r1)
        self.r0.setChecked(True)
        Afferences_choice_l = QHBoxLayout()
        Afferences_choice_l.addWidget(self.r0)
        Afferences_choice_l.addWidget(self.r1)
        self.Connection_PB = QPushButton('See Connection number matrix')
        Afferences_GB_l.addLayout(Afferences_choice_l)
        Afferences_GB_l.addWidget(self.Afferences_PB)
        Afferences_GB_l.addWidget(self.Connection_PB)
        self.Afferences_PB.clicked.connect(self.update_connections)
        self.Connection_PB.clicked.connect(self.See_connections)
        self.r0.toggled.connect(self.update_connections_per_fixed)
        self.r1.toggled.connect(self.update_connections_per_fixed)
        self.Afferences_GB.setContentLayout(Afferences_GB_l)
        self.Afferences_GB.toggleButton.click()

        # cell placement
        self.cell_placement_GB = Spoiler(title=r"cell placement")
        self.cell_placement_CB = QComboBox()
        list = ['Cylinder', 'Square', 'Rectange', 'Cylinder with curvature', 'Square with curvature',
                'Rectange with curvature']
        self.cell_placement_CB.addItems(list)
        self.cell_placement_PB = QPushButton('Place cells')
        self.cell_connectivity_PB = QPushButton('Compute connectivity')
        self.cell_keep_model_param_CB = QCheckBox('Keep model parameters')
        self.cell_keep_model_param_CB.setChecked(True)
        layoutseed = QHBoxLayout()
        self.seed_place = LineEdit_Int('0')
        layoutseed.addWidget(QLabel('Place Seed'))
        layoutseed.addWidget(self.seed_place)
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(self.cell_placement_CB, 0, 0, 1, 1)
        grid.addWidget(self.cell_placement_PB, 0, 1, 1, 2)
        grid.addWidget(self.cell_connectivity_PB, 1, 1, 1, 2)
        grid.addLayout(layoutseed, 1, 0, 1, 1)
        self.cell_placement_GB.setContentLayout(grid)
        self.cell_placement_PB.clicked.connect(self.PlaceCell_func)
        self.cell_connectivity_PB.clicked.connect(self.connectivityCell_func)

        self.EField_parameters_GB = Spoiler(title=r"E-Field parameters")
        self.EField_Conv_PB = QPushButton('Convert a txt E-Field')
        self.EField_file_PB = QPushButton('...')
        self.EField_file_TE = QLineEdit('')

        self.EField_TranslationX_LE = LineEdit('0')
        self.EField_TranslationY_LE = LineEdit('0')
        self.EField_TranslationZ_LE = LineEdit('0')
        lay_Trans = QHBoxLayout()
        lay_Trans.addWidget(QLabel('Translation'))
        lay_Trans.addWidget(QLabel('x'))
        lay_Trans.addWidget(self.EField_TranslationX_LE)
        lay_Trans.addWidget(QLabel('y'))
        lay_Trans.addWidget(self.EField_TranslationY_LE)
        lay_Trans.addWidget(QLabel('z'))
        lay_Trans.addWidget(self.EField_TranslationZ_LE)

        self.EField_RotationXY_LE = LineEdit('0')

        lay_Rot = QHBoxLayout()
        lay_Rot.addWidget(QLabel('Rotation'))
        lay_Rot.addWidget(QLabel('xy'))
        lay_Rot.addWidget(self.EField_RotationXY_LE)
        lay_Rot.addWidget(QLabel('Not Implemented yet'))

        self.EField_OnOff_CB = QCheckBox('On-Off')
        # self.EField_OnOff_CB.setEnabled(False)
        self.EField_Start_LE = LineEdit('0')
        self.EField_Length_LE = LineEdit('1000')

        self.EField_group = QButtonGroup(self)
        self.EField_File_RB = QRadioButton("Use File EField")
        self.EField_Const_RB = QRadioButton("Use Constant EField")
        self.EField_group.addButton(self.EField_File_RB)
        self.EField_group.addButton(self.EField_Const_RB)
        self.EField_Const_RB.setChecked(True)

        self.EField_Display_PB = QPushButton('Display')

        layu = QHBoxLayout()
        self.EField_Const_Ex = LineEdit('0')
        self.EField_Const_Ey = LineEdit('0')
        self.EField_Const_Ez = LineEdit('1')
        layu.addWidget(QLabel('Constant EField (mV/m)'))
        layu.addWidget(QLabel('Ex'))
        layu.addWidget(self.EField_Const_Ex)
        layu.addWidget(QLabel('Ey'))
        layu.addWidget(self.EField_Const_Ey)
        layu.addWidget(QLabel('Ez'))
        layu.addWidget(self.EField_Const_Ez)

        lay_stim = QHBoxLayout()
        self.EField_StimSig_CB = QComboBox()
        self.EField_StimSig_CB.addItems(['Constant', 'Sinusoidal', 'rectangular', 'triangular'])
        self.EField_StimSig_A_LE = LineEdit('1')
        self.EField_StimSig_F_LE = LineEdit('1')
        lay_stim.addWidget(self.EField_StimSig_CB)
        lay_stim.addWidget(QLabel('A'))
        lay_stim.addWidget(self.EField_StimSig_A_LE)
        lay_stim.addWidget(QLabel('F'))
        lay_stim.addWidget(self.EField_StimSig_F_LE)

        grid = QGridLayout()
        line = 0
        grid.addWidget(self.EField_Conv_PB, line, 1, 1, 3)
        line += 1
        grid.addWidget(QLabel('E-Field File (.mat)'), line, 0, 1, 2)
        grid.addWidget(self.EField_file_PB, line, 2, 1, 1)
        grid.addWidget(self.EField_file_TE, line, 3, 1, 2)
        line += 1

        grid.addWidget(self.EField_File_RB, line, 0, 1, 2)
        grid.addWidget(self.EField_Const_RB, line, 2, 1, 3)
        grid.addWidget(self.EField_Display_PB, line, 5, 1, 1)
        line += 1
        grid.addLayout(layu, line, 0, 1, 6)
        line += 1
        grid.addWidget(self.EField_OnOff_CB, line, 0, 1, 1)
        grid.addWidget(QLabel('Start'), line, 1, 1, 1)
        grid.addWidget(self.EField_Start_LE, line, 2, 1, 1)
        grid.addWidget(QLabel('Length'), line, 3, 1, 1)
        grid.addWidget(self.EField_Length_LE, line, 4, 1, 1)
        line += 1
        grid.addLayout(lay_stim, line, 0, 1, 4)

        self.EField_parameters_GB.setContentLayout(grid)
        self.EField_parameters_GB.toggleButton.click()

        self.EField_Conv_PB.clicked.connect(self.EField_Conv_Fun)
        self.EField_file_PB.clicked.connect(self.get_Efield_path)
        self.EField_Display_PB.clicked.connect(self.EField_Display_Fun)

        # Stim param I_inj=60, tau=4, stimDur=3, nbstim=5, varstim=12
        self.Stimulation_parameters_GB = Spoiler(title=r"Stimulation parameters")
        label1 = QLabel('Stim duration (ms)')
        label2 = QLabel(u"Injected current density (\u00B5A/cm<sup>2</sup>)")
        label3 = QLabel("Time constant RC (ms)")
        label4 = QLabel("Stim number per cell")
        label5 = QLabel("Jitter variance (ms)")
        label6 = QLabel("seed")
        self.StimDuration_e = LineEdit('3')
        self.i_inj_e = LineEdit('60')
        self.tau_e = LineEdit('4')
        self.nbStim_e = LineEdit_Int('5')
        self.varianceStim_e = LineEdit('12')
        self.seed_e = LineEdit('0')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0)
        grid.addWidget(self.StimDuration_e, 0, 1)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(self.i_inj_e, 1, 1)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(self.tau_e, 2, 1)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(self.nbStim_e, 3, 1)
        grid.addWidget(label5, 4, 0)
        grid.addWidget(self.varianceStim_e, 4, 1)
        grid.addWidget(label6, 5, 0)
        grid.addWidget(self.seed_e, 5, 1)
        self.Stimulation_parameters_GB.setContentLayout(grid)

        # InputTh param I_inj=25, tau=4, stimDur=3, nbstim=5, deltamin=14, delta=18)
        self.InputTh_parameters_GB = Spoiler(title=r"InputTh parameters")
        label1 = QLabel('Stim duration (ms)')
        label2 = QLabel(u"Injected current density (\u00B5A/cm<sup>2</sup>)")
        label3 = QLabel("Time constant RC (ms)")
        label4 = QLabel("Stim number per cell")
        label5 = QLabel("deltamin")
        label6 = QLabel("delta")
        self.TH_StimDuration_e = LineEdit('3')
        self.TH_i_inj_e = LineEdit('25')
        self.TH_tau_e = LineEdit('4')
        self.TH_nbStim_e = LineEdit_Int('5')
        self.TH_deltamin_e = LineEdit('14')
        self.TH_delta_e = LineEdit('18')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0)
        grid.addWidget(self.TH_StimDuration_e, 0, 1)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(self.TH_i_inj_e, 1, 1)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(self.TH_tau_e, 2, 1)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(self.TH_nbStim_e, 3, 1)
        grid.addWidget(label5, 4, 0)
        grid.addWidget(self.TH_deltamin_e, 4, 1)
        grid.addWidget(label6, 5, 0)
        grid.addWidget(self.TH_delta_e, 5, 1)
        self.InputTh_parameters_GB.setContentLayout(grid)

        # Simulation parameters
        self.Simulation_parameters_GB = Spoiler(title=r"Simulation parameters")
        label1 = QLabel('Simulation duration (ms)')
        label2 = QLabel(r"Sampling frequency (kHz)")
        self.SimDuration_e = LineEdit('100')
        self.Fs_e = LineEdit('25')
        self.StimStart_e = LineEdit('50')
        self.StimStop_e = LineEdit('1000')
        self.Stimtype_CB = QComboBox()
        self.Stimtype_CB.addItems(['One shot','Periodic','One shot + Uniform Th','Periodic + Uniform Th','Uniform Th + Uniform CD'])
        self.StimPeriode_e = LineEdit('200')
        self.StimJitter_e = LineEdit('0')
        self.UpdateModel_PB = QPushButton('Update Model')
        self.ModifyModel_PB = QPushButton('All Model Param')
        self.ModifyXModel_PB = QPushButton('Modify X Models')
        self.Reset_states_PB = QPushButton('Reset states')
        self.Save_states_PB = QPushButton('save States')
        self.Run_PB = QPushButton('Run')
        self.displaycurves_CB = QCheckBox('Display Curves')
        self.displayVTK_CB = QCheckBox('Display VTK')
        self.displaycurve_per_e = LineEdit('30')
        displaycurve_per_l = QLabel(r'% to plot')

        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0, 1, 2)
        grid.addWidget(self.SimDuration_e, 0, 2)
        grid.addWidget(label2, 1, 0, 1, 2)
        grid.addWidget(self.Fs_e, 1, 2)
        grid.addWidget(self.UpdateModel_PB, 0, 3)
        grid.addWidget(self.ModifyModel_PB, 1, 3)
        grid.addWidget(self.ModifyXModel_PB, 2, 3)
        grid.addWidget(self.Run_PB, 3, 3)
        grid.addWidget(self.Reset_states_PB, 4, 3)
        grid.addWidget(self.displaycurves_CB, 4, 0)
        grid.addWidget(self.displayVTK_CB, 3, 0)
        grid.addWidget(self.Save_states_PB, 5, 3)
        label = QLabel('Stim Start')
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.Stimtype_CB, 2, 0)
        label2 = QLabel('Periode')
        label2.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label2, 2, 1)
        grid.addWidget(self.StimPeriode_e, 2, 2)
        grid.addWidget(label, 3, 1)
        grid.addWidget(self.StimStart_e, 3, 2)
        grid.addWidget(QLabel('Stim Stop'), 4, 1)
        grid.addWidget(self.StimStop_e, 4, 2)
        grid.addWidget(QLabel('Stim Jit'), 5, 1)
        grid.addWidget(self.StimJitter_e, 5, 2)
        grid.addWidget(displaycurve_per_l, 6, 0)
        grid.addWidget(self.displaycurve_per_e, 6, 1)
        self.Simulation_parameters_GB.setContentLayout(grid)
        self.Run_PB.clicked.connect(self.simulate)
        self.UpdateModel_PB.clicked.connect(self.update_Model)
        self.ModifyModel_PB.clicked.connect(self.modify_Model)
        self.ModifyXModel_PB.clicked.connect(self.ModXNMMclicked)
        self.Reset_states_PB.clicked.connect(self.Reset_states_clicked)
        self.Save_states_PB.clicked.connect(self.Save_states_clicked)
        self.displaycurves_CB.stateChanged.connect(self.displaycurves_CB_fonc)
        self.displaycurves_CB.setChecked(True)
        self.displayVTK_CB.stateChanged.connect(self.displayVTK_CB_fonc)
        self.displayVTK_CB.setChecked(True)

        # electrode placement
        self.electrode_placement_GB = Spoiler(title=r"Recording electrode")
        label_Electrode_type = QLabel('Electrode type')
        self.choose_electrode_type = QComboBox()
        listitem = ['Disc', 'Cylinder']
        self.choose_electrode_type.addItems(listitem)
        label_x = QLabel('x y z')
        self.label_H=QLabel('H')
        self.electrode_x_e = LineEdit(str(self.electrode_pos[0]))
        self.electrode_y_e = LineEdit(str(self.electrode_pos[1]))
        self.electrode_z_e = LineEdit(str(self.electrode_pos[2]))
        self.Compute_LFP2_type_CB = QComboBox()
        self.Compute_LFP2_type_CB.addItems(["CSD"])
        self.Compute_LFP2_PB = QPushButton('LFP')
        self.Compute_LFPAll_CB = QCheckBox('All')
        self.Compute_LFPClear_CB = QCheckBox('Clear')
        self.Compute_LFPClear_CB.setChecked(True)
        self.Compute_LFPtxt_LE = QLineEdit('')
        # self.Compute_LFPAll_CB.setChecked(True)
        self.electrode_radius_e = LineEdit('50')
        self.electrode_angle1_e = LineEdit('0')
        self.electrode_angle2_e = LineEdit('0')
        self.electrode_cyl_h = LineEdit('2000')
        self.Compute_LFPDisk_PB = QPushButton('Disk LFP')
        self.Compute_LFPDiskCoated_PB = QPushButton('Coated Disk LFP')


        self.lfpplot_filtre_Active = QCheckBox('Active')
        self.lfpplot_filtre_order_l = QLabel("Order")
        self.lfpplot_filtre_order = LineEdit('1')
        self.lfpplot_filtre_FCL_l = QLabel("L(Hz)")
        self.lfpplot_filtre_FCL = LineEdit('1')
        self.lfpplot_filtre_FCH_l = QLabel("H(Hz)")
        self.lfpplot_filtre_FCH = LineEdit('1000')

        # self.Compute_LFPDisk_material_CB = QComboBox()
        # listitem = ['carbon', 'stainless_steel','gold']
        # self.Compute_LFPDisk_material_CB.addItems(listitem)
        # self.Compute_LFPDisk_coated_CB = QCheckBox('Coated')
        # self.Compute_LFPDisk_Time_LE = LineEdit('5')
        # self.Compute_LFPDiskCoated_type_CB = QComboBox()
        # listitem = ['carbon_non-coated_mean , radius = 200 µm',
        #             'gold_non-coated_mean , radius = 62.5 µm',
        #             'stainless_steel_non-coated_mean, radius = 62.5 µm',
        #             'carbon_coated_5s, radius = 200 µm',
        #             'carbon_coated_10s, radius = 200 µm',
        #             'carbon_coated_50s, radius = 200 µm',
        #             'gold_coated_5s, radius = 62.5 µm',
        #             'gold_coated_10s, radius = 62.5 µm',
        #             'gold_coated_50s, radius = 62.5 µm']
        # self.Compute_LFPDiskCoated_type_CB.addItems(listitem)
        self.Temporal_PSD_CB = QCheckBox('Temporal/PSD')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label_Electrode_type, 0, 0)
        grid.addWidget(self.choose_electrode_type,0,1)
        grid.addWidget(self.Compute_LFPtxt_LE,0,3)
        grid.addWidget(self.Compute_LFPClear_CB,0,4,1,2)
        grid.addWidget(label_x, 1, 0)
        grid.addWidget(self.electrode_x_e,1, 1)
        grid.addWidget(self.electrode_y_e, 1, 2)
        grid.addWidget(self.electrode_z_e, 1, 3)
        # grid.addWidget(QLabel('Type'), 0, 4)
        grid.addWidget(self.Compute_LFP2_type_CB, 1, 4 , 1 ,2)

        grid.addWidget(QLabel(u'r \u03B81 \u03B82'), 2, 0)
        grid.addWidget(self.electrode_radius_e, 2, 1)
        grid.addWidget(self.electrode_angle1_e, 2, 2)
        grid.addWidget(self.electrode_angle2_e, 2, 3)
        grid.addWidget(self.Compute_LFPAll_CB, 2, 4)
        grid.addWidget(self.Compute_LFP2_PB, 2, 5)
        grid.addWidget(self.label_H, 3, 0)
        grid.addWidget(self.electrode_cyl_h,3,1)


        self.lfpplot_filtre_Active = QCheckBox('Active')
        self.lfpplot_filtre_order = LineEdit_Int('1')
        self.lfpplot_filtre_FCL = LineEdit('1')
        self.lfpplot_filtre_FCH = LineEdit('1000')

        grid.addWidget(self.lfpplot_filtre_Active, 4, 0)
        grid.addWidget(QLabel("Order"), 4, 1)
        grid.addWidget(QLabel("L(Hz)"), 4, 2)
        grid.addWidget(QLabel("H(Hz)"), 4, 3)
        grid.addWidget(self.lfpplot_filtre_order, 5, 1)
        grid.addWidget(self.lfpplot_filtre_FCL, 5, 2)
        grid.addWidget(self.lfpplot_filtre_FCH, 5, 3)
        # grid.addWidget(QLabel('material'), 2, 0)
        # grid.addWidget(self.Compute_LFPDiskCoated_type_CB, 2, 0, 1, 4)
        # grid.addWidget(self.Compute_LFPDisk_material_CB, 3, 0, 1, 2)
        # grid.addWidget(self.Compute_LFPDisk_coated_CB, 3, 2, 1, 2)
        # grid.addWidget(QLabel('Time (s)'), 3, 4, 1, 1)
        # grid.addWidget(self.Compute_LFPDisk_Time_LE, 3, 5, 1, 1)
        # grid.addWidget(self.Compute_LFPDiskCoated_PB, 4, 4, 1, 2)
        grid.addWidget(self.Temporal_PSD_CB, 5, 4, 1, 2)

        self.electrode_placement_GB.setContentLayout(grid)
        [x.editingFinished.connect(self.electrode_placement_func) for x in
         [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e, self.electrode_cyl_h]]
        [x.editingFinished.connect(self.electrode_placement_func) for x in
         [self.electrode_radius_e, self.electrode_angle1_e, self.electrode_angle2_e]]
        self.Compute_LFP2_PB.clicked.connect(self.Compute_LFP_fonc)
        self.choose_electrode_type.activated.connect(self.electrode_placement_func)
        self.Compute_LFPDisk_PB.clicked.connect(self.Compute_LFP_fonc)
        self.Compute_LFPDiskCoated_PB.clicked.connect(self.Compute_LFPDiskCoated_fonc)
        self.Temporal_PSD_CB.stateChanged.connect(self.UpdateLFP)
        self.electrode_cyl_h.hide()
        self.label_H.hide()



        self.somaSize = 15.0 * 10e-3  # 15 micrometres = 15 * 10e-3 mm
        self.dendriteSize = 300.0 * 10e-3  # 800  micrometres = 400 * 10e-3 mm
        self.sigma = 33.0 * 10e-5
        self.gc = 10.e-5
        self.p = 0.15

        #Ap detection
        self.AP_detect_GB = Spoiler(title=r"Action Potential detection")
        self.AP_detect_l = QVBoxLayout()
        AP_detect_name1_tl=QLabel('Thresholding: ')
        self.AP_detect_name1_p_l = QHBoxLayout()
        AP_detect_name1_l=QLabel('Threshold')
        self.AP_detect_name1_LE = LineEdit('0')
        self.AP_detect_name1_PB = QPushButton('Apply')
        self.AP_detect_name1_p_l.addWidget(AP_detect_name1_tl)
        self.AP_detect_name1_p_l.addWidget(AP_detect_name1_l)
        self.AP_detect_name1_p_l.addWidget(self.AP_detect_name1_LE)
        self.AP_detect_name1_p_l.addWidget(self.AP_detect_name1_PB)
        AP_detect_name2_tl=QLabel('PeakDet: ')
        self.AP_detect_name2_p_l = QHBoxLayout()
        AP_detect_name2_l=QLabel('Delta')
        self.AP_detect_name2_LE = LineEdit('50')
        self.AP_detect_name2_PB = QPushButton('Apply')
        self.AP_detect_name2_p_l.addWidget(AP_detect_name2_tl)
        self.AP_detect_name2_p_l.addWidget(AP_detect_name2_l)
        self.AP_detect_name2_p_l.addWidget(self.AP_detect_name2_LE)
        self.AP_detect_name2_p_l.addWidget(self.AP_detect_name2_PB)

        self.AP_detect_save_p_l = QHBoxLayout()
        self.AP_detect_Save_PB = QPushButton('Save Aps')
        self.AP_detect_Save_PB.setFixedWidth(100)
        self.AP_detect_save_p_l.addWidget(self.AP_detect_Save_PB)
        self.AP_detect_l.addLayout(self.AP_detect_name1_p_l)
        self.AP_detect_l.addLayout(self.AP_detect_name2_p_l)
        self.AP_detect_l.addLayout(self.AP_detect_save_p_l)
        self.AP_detect_name1_PB.clicked.connect(lambda checked, s='Thresholding': self.AP_detect_Method_PB_Apply(s) )
        self.AP_detect_name2_PB.clicked.connect(lambda checked, s='PeakDet': self.AP_detect_Method_PB_Apply(s) )
        self.AP_detect_Save_PB.clicked.connect(self.AP_detect_save_fun)
        self.AP_detect_GB.setContentLayout(self.AP_detect_l)
        self.AP_detect_GB.toggleButton.click()

        # Phase polar
        self.AP_phase_GB = Spoiler(title=r"Action Potential EField Phase")
        self.AP_phase_l = QVBoxLayout()
        self.AP_phase_name1_p_l = QHBoxLayout()
        AP_phase_Nbin_tl = QLabel('Nbin: ')
        AP_phase_Lim_l = QLabel('Lim:')
        self.AP_phase_Nbin_LE = LineEdit_Int('100')
        self.AP_phase_Lim_LE = LineEdit('1')
        self.AP_phase_name1_PB = QPushButton('Apply')
        self.AP_phase_name1_p_l.addWidget(AP_phase_Nbin_tl)
        self.AP_phase_name1_p_l.addWidget(self.AP_phase_Nbin_LE)
        self.AP_phase_name1_p_l.addWidget(AP_phase_Lim_l)
        self.AP_phase_name1_p_l.addWidget(self.AP_phase_Lim_LE)


        self.AP_phase_polar_l = QHBoxLayout()
        self.AP_phase_polar_PB = QPushButton('Plot Polar Hist')
        self.AP_phase_polar_PB.setFixedWidth(150)
        self.AP_phase_polar_l.addWidget(self.AP_phase_polar_PB)

        self.AP_phase_l.addLayout(self.AP_phase_name1_p_l)
        self.AP_phase_l.addLayout(self.AP_phase_polar_l)
        self.AP_phase_polar_PB.clicked.connect(self.AP_phase_polar_fun)
        self.AP_phase_GB.setContentLayout(self.AP_phase_l)
        self.AP_phase_GB.toggleButton.click()



        self.Param_VBOX.addWidget(self.type_tissu_SP)
        self.Param_VBOX.addWidget(self.tissue_size_GB)
        self.Param_VBOX.addWidget(self.pourcentageCell_GB)
        self.Param_VBOX.addWidget(self.Lambda_S_D_GB)
        self.Param_VBOX.addWidget(self.Afferences_GB)
        self.Param_VBOX.addWidget(self.cell_placement_GB)
        self.Param_VBOX.addWidget(self.PCsubtypes_GB)
        self.Param_VBOX.addWidget(self.EField_parameters_GB)
        self.Param_VBOX.addWidget(self.Stimulation_parameters_GB)
        self.Param_VBOX.addWidget(self.InputTh_parameters_GB)
        self.Param_VBOX.addWidget(self.Simulation_parameters_GB)
        self.Param_VBOX.addWidget(self.electrode_placement_GB)
        self.Param_VBOX.addWidget(self.AP_detect_GB)
        self.Param_VBOX.addWidget(self.AP_phase_GB)
        self.Param_VBOX.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))
        # self.Param_VBOX.addLayout(QVBoxLayout( ))

        self.NewCreate_EField_view = []

    def AP_detect_save_fun(self, fileName = None):
        if fileName is None:
            fileName = QFileDialog.getSaveFileName(caption='Save Aps',
                                                   directory='',
                                                   filter="mat (*.mat)")
        A = float(self.EField_StimSig_A_LE.text())
        F = float(self.EField_StimSig_F_LE.text())
        stimOnOff = int(self.EField_OnOff_CB.isChecked())
        Start = float(self.EField_Start_LE.text())
        Length = float(self.EField_Length_LE.text())
        mdic = {"APs": self.Aps,
                "flatindex": self.flatindex,
                "List_Neurone_type": self.List_Neurone_type,
                "List_PCsubtypes": self.List_PCsubtypes,
                "A": A,
                "F": F,
                "stimOnOff": stimOnOff,
                "Start": Start,
                "Length": Length,
                }
        scipy.io.savemat(fileName[0], mdic)



    def Nb_Cell_Changed(self, s):
        if s in ["1", "23", "4", "5", "6"]:
            somme = 0
            somme += int(self.nbcellsnb1.text())
            somme += int(self.nbcellsnb23.text())
            somme += int(self.nbcellsnb4.text())
            somme += int(self.nbcellsnb5.text())
            somme += int(self.nbcellsnb6.text())
            self.nbcellsnbtotal.setText(str(int(somme)))
            self.update_cellNnumber()
        elif s == 'total':
            Nbcells = self.CC.Nbcells
            ratio = np.array([int(self.nbcellsnb1.text()), int(self.nbcellsnb23.text()), int(self.nbcellsnb4.text()),
                              int(self.nbcellsnb5.text()), int(self.nbcellsnb6.text())]).astype(float)
            ratio /= float(Nbcells)
            new_Nbcells = float(self.nbcellsnbtotal.text())
            self.nbcellsnb1.setText(str(int(new_Nbcells * ratio[0])))
            self.nbcellsnb23.setText(str(int(new_Nbcells * ratio[1])))
            self.nbcellsnb4.setText(str(int(new_Nbcells * ratio[2])))
            self.nbcellsnb5.setText(str(int(new_Nbcells * ratio[3])))
            self.nbcellsnb6.setText(str(int(new_Nbcells * ratio[4])))
            self.update_cellNnumber()


    def AP_detect_Method_PB_Apply(self,s):
        print(s)
        if s == 'Thresholding':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.flatindex):
                # if self.List_Neurone_type[v[0]][v[1]] == 1:
                array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            threshold = float(self.AP_detect_name1_LE.text())
            sigbin = array_Sigs > threshold

            # self.Aps = []
            # for i in range(sigbin.shape[0]):
            #     Ap = []
            #     for j in range(sigbin.shape[1] - 1):
            #         if sigbin[i, j] == False and sigbin[i, j + 1] == True:
            #             Ap.append(t[j])
            #     self.Aps.append(Ap)
            self.Aps = []
            for i in range(sigbin.shape[0]):
                Ap = list(np.where(np.diff(sigbin[i, :], axis=0))[0])
                self.Aps.append(Ap)

            self.mascene_LFPViewer.update_AP_detect(self.Aps, [item for sublist in self.List_Colors for item in sublist] , titre='Threshold')

            A = float(self.EField_StimSig_A_LE.text())
            F = float(self.EField_StimSig_F_LE.text())
            stimOnOff = int(self.EField_OnOff_CB.isChecked())
            Start = float(self.EField_Start_LE.text())
            Length = float(self.EField_Length_LE.text())
            self.mdic = {"APs": self.Aps,
                    "flatindex": self.flatindex,
                    "List_Neurone_type": self.List_Neurone_type,
                    "List_PCsubtypes": self.List_cellsubtypes_PYR,
                    "A": A,
                    "F": F,
                    "stimOnOff": stimOnOff,
                    "Start": Start,
                    "Length": Length,
                    # "LFP": self.LFP
                    }

        elif s == 'PeakDet':

            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            threshold = float(self.AP_detect_name2_LE.text())
            self.Aps = []
            for i, v in enumerate(self.flatindex):
                # sampls = signal.find_peaks(Sigs_dict[keys[i]],prominence =threshold)[0]
                sampls = peakdet(Sigs_dict[keys[i]], threshold)[0]
                if not np.array_equal(sampls, np.array([]))  :
                    self.Aps.append(t[sampls[:, 0].astype(int)].tolist())
                else:
                    self.Aps.append([])
            self.mascene_LFPViewer.update_AP_detect(self.Aps, [item for sublist in self.List_Colors for item in sublist],
                                                        titre='Find peaks')

            A = float(self.EField_StimSig_A_LE.text())
            F = float(self.EField_StimSig_F_LE.text())
            stimOnOff = int(self.EField_OnOff_CB.isChecked())
            Start = float(self.EField_Start_LE.text())
            Length = float(self.EField_Length_LE.text())
            self.mdic = {"APs": self.Aps,
                    "flatindex": self.flatindex,
                    "List_Neurone_type": self.List_Neurone_type,
                    "List_PCsubtypes": self.List_PCsubtypes,
                    "A": A,
                    "F": F,
                    "stimOnOff": stimOnOff,
                    "Start": Start,
                    "Length": Length,
                    "LFP": self.LFP
                    }




    def AP_phase_polar_fun(self):

        Nbin = int(self.AP_phase_Nbin_LE.text())
        Lim = float(self.AP_phase_Lim_LE.text())

        A = float(self.EField_StimSig_A_LE.text())
        F = float(self.EField_StimSig_F_LE.text())


        Fs = float(self.EField_StimSig_F_LE.text())

        # t = np.arange(self.nbEch) *(self.dt)/1000

        # phase = 2 * np.pi * F * t
        # phase = np.mod(phase, np.pi * 2)

        Phase_list = [np.mod(2 * np.pi * F * np.array(aps)/1000 , np.pi * 2)  for aps in self.Aps]
        Phase_flat = np.concatenate(Phase_list).ravel()

        self.mascene_LFPViewer.update_AP_polar(Phase_flat,Nbin=Nbin, Lim=Lim,  titre='')



    @staticmethod
    def _victor_purpura_dist_for_st_pair_intuitive(train_a, train_b, q=1.0):
        nspk_a = len(train_a)
        nspk_b = len(train_b)
        scr = np.zeros((nspk_a + 1, nspk_b + 1))
        scr[:, 0] = range(0, nspk_a + 1)
        scr[0, :] = range(0, nspk_b + 1)

        if nspk_a > 0 and nspk_b > 0:
            for i in range(1, nspk_a + 1):
                for j in range(1, nspk_b + 1):
                    scr[i, j] = min(scr[i - 1, j] + 1, scr[i, j - 1] + 1)
                    scr[i, j] = min(scr[i, j], scr[i - 1, j - 1]
                                    + np.float64((q * abs(train_a[i - 1] - train_b[j - 1]))))
        return scr[nspk_a, nspk_b]

    def _summed_dist_matrix(self, spiketrains, tau, presorted=False):
        # The algorithm underlying this implementation is described in
        # Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
        # Rossum distances. Network: Computation in Neural Systems, 23(1-2),
        # 48-58. We would like to remark that in this paper in formula (9) the
        # left side of the equation should be divided by two.
        #
        # Given N spiketrains with n entries on average the run-time complexity is
        # O(N^2 * n). O(N^2 + N * n) memory will be needed.

        if len(spiketrains) <= 0:
            return np.zeros((0, 0))

        if not presorted:
            spiketrains = [v.copy() for v in spiketrains]
            for v in spiketrains:
                v.sort()

        sizes = np.asarray([v.size for v in spiketrains])
        values = np.empty((len(spiketrains), max(1, sizes.max())))
        values.fill(np.nan)
        for i, v in enumerate(spiketrains):
            if v.size > 0:
                values[i, :v.size] = \
                    (v / tau)

        exp_diffs = np.exp(values[:, :-1] - values[:, 1:])
        markage = np.zeros(values.shape)
        for u in range(len(spiketrains)):
            markage[u, 0] = 0
            for i in range(sizes[u] - 1):
                markage[u, i + 1] = (markage[u, i] + 1.0) * exp_diffs[u, i]

        # Same spiketrain terms
        D = np.empty((len(spiketrains), len(spiketrains)))
        D[np.diag_indices_from(D)] = sizes + 2.0 * np.sum(markage, axis=1)

        # Cross spiketrain terms
        for u in range(D.shape[0]):
            all_ks = np.searchsorted(values[u], values, 'left') - 1
            for v in range(u):
                js = np.searchsorted(values[v], values[u], 'right') - 1
                ks = all_ks[v]
                slice_j = np.s_[np.searchsorted(js, 0):sizes[u]]
                slice_k = np.s_[np.searchsorted(ks, 0):sizes[v]]
                D[u, v] = np.sum(
                    np.exp(values[v][js[slice_j]] - values[u][slice_j]) *
                    (1.0 + markage[v][js[slice_j]]))
                D[u, v] += np.sum(
                    np.exp(values[u][ks[slice_k]] - values[v][slice_k]) *
                    (1.0 + markage[u][ks[slice_k]]))
                D[v, u] = D[u, v]

        return D

    @staticmethod
    @njit
    def ISI_ratio(array_ISI, array_ratio):
        for i in range(array_ISI.shape[0]):  # xisi
            for j in range(array_ISI.shape[0]):  # y
                if not j > i:
                    for tt in range(array_ISI.shape[1]):
                        if array_ISI[i, tt] <= array_ISI[j, tt]:
                            array_ratio[i, j, tt] = array_ISI[i, tt] / array_ISI[j, tt] - 1
                            array_ratio[j, i, tt] = array_ISI[i, tt] / array_ISI[j, tt] - 1
                        else:
                            array_ratio[i, j, tt] = -(array_ISI[j, tt] / array_ISI[i, tt] - 1)
                            array_ratio[j, i, tt] = -(array_ISI[j, tt] / array_ISI[i, tt] - 1)
        return array_ratio

    def get_new_List_C(self):
        for l in range(len(self.List_C)):
            for c in range(len(self.List_C[0])):
                self.List_C[l, c] = float(self.List_C_e[l * len(self.List_C[0]) + c].text())
                # if c == 4:
                #     self.List_C[l, c] = float(self.List_C_e[l*len(self.List_C[0])+c].text() )
                # else:
                #     self.List_C[l, c] = int(float(self.List_C_e[l*len(self.List_C[0])+c].text() ))

    def get_new_List_Var(self):
        for l in range(len(self.List_Var)):
            for c in range(len(self.List_Var)):
                self.List_Var[l, c] = float(self.List_Var_e[l * len(self.List_Var) + c].text())

    pyqtSlot(int)

    def PlaceCell_msg(self, cellnb):
        if cellnb == -2:
            self.msg = msg_wait(
                "Cell Placement in progress\n" + '0/' + str(np.sum(self.CC.Layer_nbCells)) + "\nPlease wait.")
            self.msg.setStandardButtons(QMessageBox.StandardButton.Cancel)
            self.PlaceCell_msg_cnacel = False
            self.parent.processEvents()
        elif cellnb == -1:
            self.PlaceCell_msg_cnacel = False
        else:
            self.msg.setText("Cell Placement in progress\n" + str(cellnb) + '/' + str(
                np.sum(self.CC.Layer_nbCells)) + "\nPlease wait.")
            self.parent.processEvents()

    def connectivityCell_func(self):
        seed = int(self.seed_place.text())
        print(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR, self.CC.NB_Th, self.CellPosition,seed)
        self.CC.Conx = Connectivity.Create_Connectivity_Matrix(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR,
                                                               self.CC.NB_Th,
                                                               self.CellPosition,
                                                               seed=seed)
        self.update_model_with_same_param()
        self.masceneCM.update()

    def EField_Conv_Fun(self):
        fileName = QFileDialog.getOpenFileName(self, "E-Field text file", "", "txt (*.txt)")
        if fileName[0] == '':
            return
        if fileName[1] == "txt (*.txt)":
            try:
                n = np.loadtxt(fileName[0], comments='%')

                x = n[:, 0] * 1000
                xmin = np.round(np.min(x) * 1000000) / 1000000
                xmax = np.round(np.max(x) * 1000000) / 1000000
                xsteps = len(set(x))
                xrange = abs(xmax - xmin)
                y = n[:, 1] * 1000
                ymin = np.round(np.min(y) * 1000000) / 1000000
                ymax = np.round(np.max(y) * 1000000) / 1000000
                ysteps = len(set(y))
                yrange = abs(ymax - ymin)
                z = n[:, 2] * 1000
                zmin = np.round(np.min(z) * 1000000) / 1000000
                zmax = np.round(np.max(z) * 1000000) / 1000000
                zsteps = len(set(z))
                zrange = abs(zmax - zmin)

                E = n[:, 4:] * 1000
                Er = np.zeros((xsteps, ysteps, zsteps, 3), dtype=float)
                i = 0
                for kx in range(xsteps):
                    for ky in range(ysteps):
                        for kz in range(zsteps):
                            Er[kx, ky, kz, :] = E[i, :]
                            i += 1

                mdic = {"Er": Er,
                        "xmin": xmin,
                        "xmax": xmax,
                        "xsteps": xsteps,
                        "xrange": xrange,
                        "ymin": ymin,
                        "ymax": ymax,
                        "ysteps": ysteps,
                        "yrange": yrange,
                        "zmin": zmin,
                        "zmax": zmax,
                        "zsteps": zsteps,
                        "zrange": zrange, }

                fileName = QFileDialog.getSaveFileName(caption='Save Efield',
                                                       directory=os.path.splitext(fileName[0])[0] + '.mat',
                                                       filter="mat (*.mat)")
                if (fileName[0] == ''):
                    return
                if os.path.splitext(fileName[0])[1] == '':
                    fileName = (fileName[0] + '.mat', fileName[1])
                if fileName[1] == "mat (*.mat)":
                    scipy.io.savemat(fileName[0], mdic)
            except:
                msg_cri(s='Something went wrong')

    def get_Efield_path(self, ):
        extension = "mat"
        fileName = QFileDialog.getOpenFileName(caption='Load mesh file', filter=extension + " (*." + extension + ")")
        if (fileName[0] == ''):
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName = (fileName[0] + '.' + extension, fileName[1])
        if fileName[1] == extension + " (*." + extension + ")":
            try:
                self.EField_file_TE.setText(fileName[0])
                self.EField_Load_file()
            except:
                self.EField_file_TE.setText('')
                msg_cri('VTK file cannot be loaded.\nPlease, load another one.')

    def EField_Load_file(self):
        mat = scipy.io.loadmat(self.EField_file_TE.text())
        self.EFieldFile = {}
        for key in [key for key in mat.keys() if not '__' in key]:
            self.EFieldFile[key] = mat[key]  # [0][0]
            if self.EFieldFile[key].shape == (1, 1):
                self.EFieldFile[key] = self.EFieldFile[key][0][0]

        self.EField_XYZ_comp()
        self.EField_Rot_XY_comp()
        self.EFieldFile['Er'] = self.EFieldFile['Er'].astype(float)

        self.EField_OnOff_CB.setEnabled(True)

    def EField_XYZ_comp(self):
        xT = float(self.EField_TranslationX_LE.text())
        yT = float(self.EField_TranslationY_LE.text())
        zT = float(self.EField_TranslationZ_LE.text())

        self.EFieldFile['x'] = np.linspace(-self.EFieldFile['xrange'] / 2 + xT, self.EFieldFile['xrange'] / 2 + xT,
                                           self.EFieldFile['xsteps'])
        self.EFieldFile['y'] = np.linspace(-self.EFieldFile['yrange'] / 2 + yT, self.EFieldFile['yrange'] / 2 + yT,
                                           self.EFieldFile['ysteps'])
        self.EFieldFile['z'] = np.linspace(0 + zT, self.EFieldFile['zrange'] + zT, self.EFieldFile['zsteps'])

    def EField_Rot_XY_comp(self):
        self.EField_theta = float(self.EField_RotationXY_LE.text())
        # sintheta = np.sin(theta)
        # costheta = np.cos(theta)
        # for ix in range(self.EFieldFile['Er'].shape[0]):
        #     x = self.EFieldFile['x'][ix]
        #     for iy in range(self.EFieldFile['Er'].shape[1]):
        #         y = self.EFieldFile['y'][iy]
        #         for iz in range(self.EFieldFile['Er'].shape[2]):
        #             x2 = self.EFieldFile['Er'][ix,iy,iz,0] + x
        #             y2 = self.EFieldFile['Er'][ix,iy,iz,1] + y
        #             xp = x2 * costheta - y2 * sintheta
        #             yp = x2 * sintheta + y2 * costheta
        #             self.EFieldFile['Er'][ix,iy,iz,0] = xp
        #             self.EFieldFile['Er'][ix,iy,iz,1] = yp
        # x = self.EFieldFile['x']
        # b = np.repeat(x[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # b2 = np.repeat(b[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        # y = self.EFieldFile['y']
        # c = np.repeat(y[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # c2 = np.repeat(c[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        #
        # x2 = self.EFieldFile['Er'][:,:,:,0] + b2
        # y2 = self.EFieldFile['Er'][:,:,:,1] + c2
        # xp = x2 * costheta - y2 * sintheta
        # yp = x2 * sintheta + y2 * costheta
        #
        # self.EFieldFile['Er'][:,:,:,0] = xp
        # self.EFieldFile['Er'][:,:,:,1] = yp

        # x = self.EFieldFile['x']
        # y = self.EFieldFile['y']
        # xp = x * np.cos(theta) - 0 * np.sin(theta)
        # yp = 0 * np.sin(theta) + y * np.cos(theta)
        # self.EFieldFile['x'] = xp
        # self.EFieldFile['y'] = yp

        # x = self.EFieldFile['x']
        # b = np.repeat(x[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # b2 = np.repeat(b[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        # y = self.EFieldFile['y']
        # c = np.repeat(y[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # c2 = np.repeat(c[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        #
        # self.EFieldFile['Er'][:,:,:,0] -= b2
        # self.EFieldFile['Er'][:,:,:,1] -= c2

    def EField_Const_fun(self):
        Er = np.array([float(self.EField_Const_Ex.text()), float(self.EField_Const_Ey.text()),
                       float(self.EField_Const_Ez.text())])
        self.EField_Const = {"Er": Er}

    def EField_Display_Fun(self):
        if self.EField_Const_RB.isChecked():
            self.EField_Const_fun()
            self.EField = self.EField_Const
            self.NewCreate_EField_view.append(Graph_EField_VTK(self, fromfile=False))
            self.NewCreate_EField_view[-1].show()

        elif self.EField_File_RB.isChecked():
            self.EField_Load_file()
            self.EField = self.EFieldFile
            self.NewCreate_EField_view.append(Graph_EField_VTK(self))
            self.NewCreate_EField_view[-1].show()

    # except:
    #     msg_cri("Display not available")
    def updateMSGfromfunction(self,text=''):
        self.msg.setText(text)
        self.parent.processEvents()

    def PlaceCell_func(self):
        self.msg = msg_wait("Computation in progress\nPlease wait.")
        self.msg.setStandardButtons(QMessageBox.StandardButton.Cancel)
        # self.msg.buttonClicked.connect(self.Cancelpressed)
        self.parent.processEvents()

        seed = int(self.seed_place.text())
        placement = self.cell_placement_CB.currentText()
        self.updateMSGfromfunction('cell placement')
        self.CC.updateCell.something_happened.connect(self.PlaceCell_msg)
        self.CellPosition = CreateColumn.PlaceCell_func(self.CC.L, self.CC.Layer_d, self.CC.D, self.CC.Layer_nbCells,
                                                        placement, seed=seed, C=self.CC.Curvature)
        self.CellPosition = np.array(self.CellPosition)




        self.updateMSGfromfunction('Create_Connectivity_Matrices')
        t0 = time.time()
        # print(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR, self.CC.NB_Th, self.CellPosition,seed)
        self.CC.Conx = Connectivity.Create_Connectivity_Matrix(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR,
                                                               self.CC.NB_Th,
                                                               self.CellPosition, seed=seed, func=self.updateMSGfromfunction)
        # x, y = Connectivity.Computeinputdistribution(self.CC.Conx['PreSynapticPos_AMPA'],
        #                                              self.CC.Conx['PreSynapticPos_GABA'], self.CC.C)

        print(time.time() - t0)
        self.CC.updateCell.something_happened.disconnect(self.PlaceCell_msg)

        self.createCells = True

        self.updateMSGfromfunction('Updating Model')
        # # self.Create_Connectivity_Matrices()
        # # print(self.ConnectivityMatrix)
        # print('List_Names')
        self.List_Names = []
        self.List_Colors = []
        self.List_Neurone_type = []

        colors_PYR = ['#000000', '#9370db', '#9400d3', '#8b008b', '#4b0082']
        for i in range(5):
            s = ''
            if i == 0:
                s = s + ("l1_")
            elif i == 1:
                s = s + ("l23_")
            elif i == 2:
                s = s + ("l4_")
            elif i == 3:
                s = s + ("l5_")
            elif i == 4:
                s = s + ("l6_")
            Names = []
            Colors = []
            Neurone_type = []
            for j in range(self.CC.C.NB_PYR[i]):
                Names.append(s + 'PYR_' + str(j))
                Colors.append(colors_PYR[i])
                Neurone_type.append(1)

            for j in range(self.CC.C.NB_PV[i]):
                Names.append(s + 'PV_' + str(j))
                Colors.append('#228B22')
                Neurone_type.append(2)

            for j in range(self.CC.C.NB_SST[i]):
                Names.append(s + 'SST_' + str(j))
                Colors.append('#0000cd')
                Neurone_type.append(3)

            for j in range(self.CC.C.NB_VIP[i]):
                Names.append(s + 'VIP_' + str(j))
                Colors.append('#cd5c5c')
                Neurone_type.append(4)

            for j in range(self.CC.C.NB_RLN[i]):
                Names.append(s + 'RLN_' + str(j))
                Colors.append('#FFA500')
                Neurone_type.append(5)

            self.List_Colors.append(Colors)
            self.List_Names.append(Names)
            self.List_Neurone_type.append(Neurone_type)

        self.update_Model()
        self.electrode_placement_func()
        self.msg.close()
        self.masceneCM.update()
        self.update_graph()

    def scalesize_func(self):
        exPopup = Rescalesize(self, x=self.x_e.text(), y=self.y_e.text(), z=self.z_e.text())
        # exPopup.show()
        if exPopup.exec() == QDialog.DialogCode.Accepted:
            # exPopup.editfromlabel()
            # self.parent.Graph_Items[cellId] = exPopup.item
            print('Accepted')
            xs = float(exPopup.xs_e.text())
            ys = float(exPopup.ys_e.text())
            zs = float(exPopup.zs_e.text())
            self.x_e.setText(str(float(self.x_e.text()) * xs))
            self.y_e.setText(str(float(self.y_e.text()) * ys))
            self.z_e.setText(str(float(self.z_e.text()) * zs))

            self.CellPosition = self.CellPosition * np.array([xs, ys, zs])

            self.update_graph()


        else:
            print('Cancelled')
        exPopup.deleteLater()

    def update_graph(self):
        if self.displayVTK_CB.isChecked():
            self.Graph_viewer.draw_Graph()
            self.Graph_viewer.set_center()
            self.parent.processEvents()

    def electrode_placement_func(self,update = True):
        if self.choose_electrode_type.currentText()=='Cylinder':
            self.electrode_cyl_h.show()
            self.label_H.show()
        else:
            self.electrode_cyl_h.hide()
            self.label_H.hide()

        self.electrode_pos = [float(x.text()) for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]

        if float(self.electrode_radius_e.text()) == 0.:
            self.electrode_disk = [0,
                                   float(self.electrode_radius_e.text()),
                                   -float(self.electrode_angle1_e.text()),
                                   -float(self.electrode_angle2_e.text()),
                                   float(self.electrode_cyl_h.text())]
        else:

            self.electrode_disk = [1,
                                   float(self.electrode_radius_e.text()),
                                   -float(self.electrode_angle1_e.text()),
                                   -float(self.electrode_angle2_e.text()),
                                   float(self.electrode_cyl_h.text())]
        if update:
            self.update_graph()

    def bruitGaussien(self, s, m):
        return np.random.normal(m, s)

    def Compute_synaptic_connections_sparse(self):
        self.PreSynaptic_Cell_AMPA = []
        self.PreSynaptic_Cell_GABA = []
        self.PreSynaptic_Soma_Dend_AMPA = []
        self.PreSynaptic_Soma_Dend_AMPA_not = []
        self.PreSynaptic_Soma_Dend_GABA = []
        self.PreSynaptic_Soma_Dend_GABA_not = []
        for i, c in enumerate(self.List_Neurone_type):
            # convect = self.ConnectivityMatrix[i, :]
            # convect = self.ConnectivityMatrix.getrow(i).todense()
            # convect = np.where(convect == 1)[0]
            convect = self.ConnectivityMatrix[i]
            convect_AMPA = []
            convect_GABA = []
            convect_Soma_Dend_AMPA = []
            convect_Soma_Dend_GABA = []
            for k in convect:
                if self.List_Neurone_type[k] in [1, 2]:  # si from CA1 ou CA3
                    convect_AMPA.append(k)
                    if c in [3, 4, 5]:  # interneurones
                        convect_Soma_Dend_AMPA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # si from CA1, CA3, SOM, ou BIS
                            convect_Soma_Dend_AMPA.append(0)
                        else:  # si from BAS
                            convect_Soma_Dend_AMPA.append(1)

                else:  # from interneurone
                    convect_GABA.append(k)
                    if c in [3, 4, 5]:  # interneurones
                        convect_Soma_Dend_GABA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # si from CA1, CA3, SOM, ou BIS
                            convect_Soma_Dend_GABA.append(0)
                        else:  # si from BAS
                            convect_Soma_Dend_GABA.append(1)

            self.PreSynaptic_Cell_AMPA.append(convect_AMPA)
            self.PreSynaptic_Cell_GABA.append(convect_GABA)
            self.PreSynaptic_Soma_Dend_AMPA.append(np.array(convect_Soma_Dend_AMPA))
            self.PreSynaptic_Soma_Dend_AMPA_not.append(np.abs(np.array(convect_Soma_Dend_AMPA) - 1))
            self.PreSynaptic_Soma_Dend_GABA.append(np.array(convect_Soma_Dend_GABA))
            self.PreSynaptic_Soma_Dend_GABA_not.append(np.abs(np.array(convect_Soma_Dend_GABA) - 1))

    def update_lambda(self):
        if self.ld0.isChecked():
            self.CC.var_lambdas=0

        elif self.ld1.isChecked():
            self.CC.var_lambdas=1




        pass
        # self.CC.updateLambda()

    def update_PCsubtypes(self):
        i=0
        for l in range(self.CC.C.PCsubtypes_Percentage.shape[0]):
            for c in range(self.CC.C.PCsubtypes_Percentage.shape[1]):
                self.CC.C.PCsubtypes_Percentage[l,c] = float(self.List_PCsubtypes[i].text())
                i += 1
        self.CC.C.update_morphology()

    def update_cellNnumber(self):

        Layer_nbCells = np.array(
            [int(self.nbcellsnb1.text()), int(self.nbcellsnb23.text()), int(self.nbcellsnb4.text()),
             int(self.nbcellsnb5.text()), int(self.nbcellsnb6.text())])

        PYRpercent = []
        for e in self.List_PYRpercent:
            PYRpercent.append(float(e.text()))
        PYRpercent = np.array(PYRpercent)

        PVpercent = []
        for e in self.List_PVpercent:
            PVpercent.append(float(e.text()))
        PVpercent = np.array(PVpercent)

        SSTpercent = []
        for e in self.List_SSTpercent:
            SSTpercent.append(float(e.text()))
        SSTpercent = np.array(SSTpercent)

        VIPpercent = []
        for e in self.List_VIPpercent:
            VIPpercent.append(float(e.text()))
        VIPpercent = np.array(VIPpercent)

        RLNpercent = []
        for e in self.List_RLNpercent:
            RLNpercent.append(float(e.text()))
        RLNpercent = np.array(RLNpercent)

        self.CC.update_cellNumber(Layer_nbCells=Layer_nbCells,
                                  PYRpercent=PYRpercent,
                                  PVpercent=PVpercent,
                                  SSTpercent=SSTpercent,
                                  VIPpercent=VIPpercent,
                                  RLNpercent=RLNpercent)

        self.nbcellsnbtotal.setText(str(int(np.sum(self.CC.Layer_nbCells))))
        self.CC.update_connections(fixed=not self.r0.isChecked())

    def update_connections(self):
        Norm = Afferences_ManagmentTable(self)
        if Norm.exec():
            self.CC.update_connections(self.CC.Afferences, fixed=not self.r0.isChecked())
            pass

    def See_connections(self):
        Norm = Connection_ManagmentTable(self)
        if Norm.exec():
            pass

    def update_connections_per_fixed(self):
        self.CC.update_connections(self.CC.Afferences, fixed=not self.r0.isChecked())

    def take_cell_number(self):
        if not int(self.Nb_of_PYR_l.text()) == self.Nb_of_PYR:
            self.createCells = True
        self.Nb_of_PYR = int(self.Nb_of_PYR_l.text())
        if not int(self.Nb_of_BAS_l.text()) == self.Nb_of_BAS:
            self.createCells = True
        self.Nb_of_BAS = int(self.Nb_of_BAS_l.text())
        if not int(self.Nb_of_SOM_l.text()) == self.Nb_of_SOM:
            self.createCells = True
        self.Nb_of_SOM = int(self.Nb_of_SOM_l.text())
        if not int(self.Nb_of_BIS_l.text()) == self.Nb_of_BIS:
            self.createCells = True
        self.Nb_of_BIS = int(self.Nb_of_BIS_l.text())
        if not int(self.Nb_of_PYR_Stimulated_l.text()) == self.Nb_of_PYR_Stimulated:
            self.createCells = True
        self.Nb_of_PYR_Stimulated = int(self.Nb_of_PYR_Stimulated_l.text())

        self.Nb_of_PYR_BAS_SOM_BIS_sum = [self.Nb_of_PYR,
                                          self.Nb_of_PYR + self.Nb_of_BAS,
                                          self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM,
                                          self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM + self.Nb_of_BIS]

        self.List_Neurone_type = [2] * self.Nb_of_PYR_Stimulated + [1] * (
                    self.Nb_of_PYR - self.Nb_of_PYR_Stimulated) + [3] * self.Nb_of_BAS + [4] * self.Nb_of_SOM + [
                                     5] * self.Nb_of_BIS
        self.List_PYR_Stimulated = [i for i in range(self.Nb_of_PYR_Stimulated)]

    def update_percent_sum(self):
        sum = float(self.Per_PYR.text()) + float(self.Per_BAS.text()) + float(self.Per_SOM.text()) + float(
            self.Per_BIS.text())
        self.labelSUM2.setText(str(sum))
        if sum == 100:
            self.labelSUM2.setStyleSheet("QLabel { background-color : none}")
        else:
            self.labelSUM2.setStyleSheet("QLabel { background-color : red}")

    def set_tissue_func(self):

        D = float(self.D_e.text())
        L = float(self.L_e.text())
        C = float(self.C_e.text())

        L_d1 = float(self.Layer_d1_l.text())
        L_d23 = float(self.Layer_d23_l.text())
        L_d4 = float(self.Layer_d4_l.text())
        L_d5 = float(self.Layer_d5_l.text())
        L_d6 = float(self.Layer_d6_l.text())

        self.CC.updateTissue(D, L, C, np.array([L_d1, L_d23, L_d4, L_d5, L_d6]))

    def Generate_Stim_Signals(self):

        self.nbOfSamplesStim = int(float(self.StimDuration_e.text()) / self.dt)
        self.i_inj = float(self.i_inj_e.text())
        self.tau = float(self.tau_e.text())
        self.nbStim = int(self.nbStim_e.text())
        self.varianceStim = int(float(self.varianceStim_e.text()) / self.dt)
        self.seed = int(self.seed_e.text())

        nb_Stim_Signals = len(self.List_PYR_Stimulated)

        self.dt = 1 / float(self.Fs_e.text())
        self.T = float(self.SimDuration_e.text())
        self.nbEch = int(self.T / self.dt)

        Stim_Signals_in = np.zeros((nb_Stim_Signals, self.nbEch))
        Stim_Signals_out = np.zeros((nb_Stim_Signals, self.nbEch))

        Generate_Stim_Signals(Stim_Signals_in, self.seed, self.nbOfSamplesStim, self.i_inj, self.tau, self.nbStim,
                              self.varianceStim, nb_Stim_Signals, self.nbEch, self.dt, Stim_Signals_out)

        self.Stim_Signals = Stim_Signals_out

        if not self.seed == 0:
            np.random.seed(self.seed)
        # else:
        #     np.random.seed()

    def update_model_with_same_param(self):
        List_Neurone_param = copy.deepcopy(self.CC.List_Neurone_param)
        self.CC.create_cells()
        self.createCells = False
        self.Reset_states_clicked()
        self.CC.List_Neurone_param = List_Neurone_param
        self.CC.Update_param_model()

    def update_Model(self):

        # self.Compute_synaptic_connections_sparse()
        self.CC.create_cells()
        self.createCells = False
        self.Reset_states_clicked()

    def modify_Model(self):
        try:
            if self.NewModify1NMM == None and self.NewModifyXNMM == None:
                self.NewModify1NMM = Modify_1_NMM(Dict_Param=self.CC.List_Neurone_param, List_Names=self.List_Names,
                                                  List_Color=self.List_Colors)
                self.NewModify1NMM.Mod_OBJ.connect(self.ApplyMod1NMM)
                self.NewModify1NMM.Close_OBJ.connect(self.closeMod1NMM)
                self.NewModify1NMM.show()
            else:
                self.put_window_on_top()
        except:
            pass

    @pyqtSlot(list, list, list)
    def ApplyMod1NMM(self, Dict_Param, popName, popColor):
        for idx, p in enumerate(Dict_Param):
            for idx_v, key in enumerate(Dict_Param[idx].keys()):
                self.CC.List_Neurone_param[idx][key] = Dict_Param[idx][key]
        # self.CC.List_Neurone_param = Dict_Param
        self.List_Names = popName
        self.List_Colors = popColor
        self.update_graph()

    @pyqtSlot()
    def closeMod1NMM(self, ):
        self.NewModify1NMM.deleteLater()
        self.NewModify1NMM.destroyed.connect(self.on_destroyed_NewModify1NMM)
        # self.NewModify1NMM = None

    @pyqtSlot('QObject*')
    def on_destroyed_NewModify1NMM(self, o):
        self.NewModify1NMM = None

    def ModXNMMclicked(self, ):
        if self.NewModify1NMM == None and self.NewModifyXNMM == None:
            self.NewModifyXNMM = Modify_X_NMM(parent=self, List_Neurone_type=self.List_Neurone_type,
                                              Dict_Param=self.CC.List_Neurone_param, List_Names=self.List_Names,
                                              List_Color=self.List_Colors, initcell=0,
                                              CellPosition=self.CellPosition)
            self.NewModifyXNMM.Mod_OBJ.connect(self.ApplyModXNMM)
            self.NewModifyXNMM.Close_OBJ.connect(self.close_ModXNMM)
            self.NewModifyXNMM.updateVTK_OBJ.connect(self.update_VTKgraph_from_ModXNMM)
            self.NewModifyXNMM.show()
        else:
            self.put_window_on_top()


    @pyqtSlot(list, list, list)
    def ApplyModXNMM(self, Dict_Param, popName, popColor):
        for idx, p in enumerate(Dict_Param):
            for idx2, p2 in enumerate(p):
                for idx_v, key in enumerate(p2.keys()):
                    self.CC.List_Neurone_param[idx][idx2][key] = Dict_Param[idx][idx2][key]
        # self.CC.List_Neurone_param = Dict_Param
        self.CC.Update_param_model()
        self.List_Names = popName
        self.List_Colors = popColor
        self.update_graph()

    @pyqtSlot()
    def close_ModXNMM(self, ):
        self.NewModifyXNMM.deleteLater()
        self.NewModifyXNMM.destroyed.connect(self.on_destroyed_NewModifyXNMM)
        # pass
        # self.NewModifyXNMM = None

    @pyqtSlot('QObject*')
    def on_destroyed_NewModifyXNMM(self, o):
        self.NewModifyXNMM = None
        print(self.NewModifyXNMM)

    @pyqtSlot(list, )
    def update_VTKgraph_from_ModXNMM(self, selectedcell):
        self.Graph_viewer.selected_cells = selectedcell
        if self.displayVTK_CB.isChecked():
            self.Graph_viewer.draw_Graph()

    def update_ModXNMM_from_VTKgraph(self, seleced_cell):
        if not self.NewModifyXNMM == None:
            self.NewModifyXNMM.PopNumber.setCurrentIndex(seleced_cell)

    def put_window_on_top(self):
        self.parent.processEvents()
        if not self.NewModify1NMM == None:
            self.NewModify1NMM.activateWindow()
        elif not self.NewModifyXNMM == None:
            self.NewModifyXNMM.activateWindow()

    def update_samples(self):
        try:
            self.CC.tps_start = 0.
            self.CC.Reset_states()
        except:
            pass

    def Reset_states_clicked(self):
        try:
            self.CC.tps_start = 0.
            self.CC.Reset_states()
            np.random.seed(None)
        except:
            pass

    def Save_states_clicked(self,Filepath = None):
        extension = "pickle"
        if not isinstance(Filepath, str):
            fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=extension + " (*." + extension + ")")
            if (fileName[0] == ''):
                return
            if os.path.splitext(fileName[0])[1] == '':
                fileName = (fileName[0] + '.' + extension, fileName[1])
        else:
            fileName = [Filepath,'pickle']
        try:
            file_pi = open(fileName[0], 'wb')
            for var in [self.t, self.pyrVs, self.pyrVd, self.pyrVa, self.PV_Vs, self.SST_Vs, self.VIP_Vs, self.RLN_Vs, self.DPYR_Vs, self.Th_Vs, self.pyrPPSE, self.pyrPPSI, self.pyrPPSI_s, self.pyrPPSI_a]:
                pickle.dump(var, file_pi, -1)
            file_pi.close()
        except:
            msg_cri('impossible to save')

    def simulate(self):
        if self.CC.ImReady == False:
            msg_cri('The model is not ready to Simulate')
            print('The model is not ready to Simulate')
            return
        # if not self.Colonne_cortical_Thread.isRunning():
        t0 = time.time()
        self.msg = msg_wait("Computation in progress\nPlease wait.")
        self.msg.setStandardButtons(QMessageBox.StandardButton.Cancel)
        # self.msg.buttonClicked.connect(self.Cancelpressed)
        self.parent.processEvents()

        Fs = int(self.Fs_e.text())
        self.dt = 1 / Fs
        self.T = float(self.SimDuration_e.text())
        self.nbEch = int(self.T / self.dt)
        self.CC.nbEch = self.nbEch
        self.CC.dt = self.dt
        # self.CC.update_samples(Fs,self.T)
        # self.Reset_states_clicked()
        S_nbOfSamplesStim = float(self.StimDuration_e.text())
        S_i_inj = float(self.i_inj_e.text())
        S_tau = float(self.tau_e.text())
        S_nbStim = int(self.nbStim_e.text())
        S_varianceStim = float(self.varianceStim_e.text())
        S_seed = int(self.seed_e.text())
        S_StimStart = float(self.StimStart_e.text())
        S_StimStop = float(self.StimStop_e.text())
        Stimtype = self.Stimtype_CB.currentText()
        StimPeriode = float(self.StimPeriode_e.text())
        Jitter = float(self.StimJitter_e.text())

        # if not S_seed == 0:
        #     np.random.seed(S_seed)
        # else:
        #     np.random.seed()
        self.CC.set_seed(S_seed)
        self.CC.T = self.T
        self.CC.Generate_Stims(I_inj=S_i_inj, tau=S_tau, stimDur=S_nbOfSamplesStim, nbstim=S_nbStim,
                               varstim=S_varianceStim, StimStart=S_StimStart,S_StimStop=S_StimStop, type=Stimtype, periode=StimPeriode, nbEch=self.nbEch,Jitter=Jitter)

        TH_nbOfSamplesStim = float(self.TH_StimDuration_e.text())
        TH_i_inj = float(self.TH_i_inj_e.text())
        TH_tau = float(self.TH_tau_e.text())
        TH_nbStim = int(self.TH_nbStim_e.text())
        TH_deltamin = float(self.TH_deltamin_e.text())
        TH_delta = float(self.TH_delta_e.text())

        self.CC.Generate_input(I_inj=TH_i_inj, tau=TH_tau, stimDur=TH_nbOfSamplesStim, nbstim=TH_nbStim,
                               deltamin=TH_deltamin, StimStart=S_StimStart, delta=TH_delta, type=Stimtype)#)




        # plt.plot(np.transpose(self.CC.Stim_Signals[0:10, :]))
        # plt.show()
        # plt.plot(np.transpose(self.CC.Stim_InputSignals[0:10, :]))
        # plt.show()



        self.update_lambda()
        if self.EField_OnOff_CB.isChecked():
            if self.EField_Const_RB.isChecked():
                self.EField_Const_fun()
                self.EField = self.EField_Const
            elif self.EField_File_RB.isChecked():
                self.EField = self.EFieldFile
            self.CC.Generate_EField(self.EField, float(self.EField_OnOff_CB.isChecked()),
                                    bool(self.EField_Const_RB.isChecked()))
        else:
            self.EField = {'Er': np.array([0., 0., 0.])}
            self.CC.Generate_EField(self.EField, 0., True)

        A = float(self.EField_StimSig_A_LE.text())
        self.F = float(self.EField_StimSig_F_LE.text())
        stimOnOff = int(self.EField_OnOff_CB.isChecked())
        Start = float(self.EField_Start_LE.text())
        Length = float(self.EField_Length_LE.text())
        ###if tCS####
        if stimOnOff==1:
            ###if tACS
            if self.EField_StimSig_CB.currentText()=='Constant':
                self.CC.stim_freq = 0.
            else:
                self.CC.stim_freq=self.F
        self.CC.Generate_EField_Stim(self.EField_StimSig_CB.currentText(), A, self.F, stimOnOff, Start=Start,
                                     Length=Length)


        self.t0 = time.time()

        self.t, self.pyrVs,  self.W_plasticity, self.pyrVd, self.pyrVa, self.PV_Vs, self.SST_Vs, self.VIP_Vs, self.RLN_Vs, self.DPYR_Vs, self.Th_Vs, self.pyrPPSE, self.pyrPPSI, self.pyrPPSI_s, self.pyrPPSI_a,self.pyrPPSE_Dpyr,self.pyrPPSE_Th,self.pyrI_S, self.pyrI_d, self.pyrI_A = self.CC.runSim()

        # for i in range(len(self.DPYR_Vs)):
        #     plt.plot(self.t,self.DPYR_Vs[i,:]*20+200*i,'k')
        #
        # plt.show()



        self.PPS = {}
        self.PPS['PPSE'] = self.pyrPPSE
        self.PPS['PPSE_Dpyr'] = self.pyrPPSE_Dpyr
        self.PPS['PPSE_Th'] = self.pyrPPSE_Th
        self.PPS['PPSI'] = self.pyrPPSI
        self.PPS['PPSI_s'] = self.pyrPPSI_s
        self.PPS['PPSI_a'] = self.pyrPPSI_a

        self.I_all={}
        self.I_all['pyrI_S']=self.pyrI_S
        self.I_all['pyrI_d']=self.pyrI_d
        self.I_all['pyrI_A']=self.pyrI_A


        self.msg = msg_wait("Computation finished\nResults are currently displayed.\nPlease wait.")


        # if 0:
        #
        #     indices = []
        #     ind = 0
        #     for j in range(5):
        #         for i in range(len(self.CC.List_celltypes[j])):
        #             if self.CC.List_celltypes[j][i] == 0:
        #                 indices.append(ind)
        #             ind += 1
        #
        #     self.mascene_LFPViewer.clearfig()
        #     index = np.random.randint(0,1000)
        #     for i in range(5):
        #         self.mascene_LFPViewer.addline(self.pyrPPSE[i,index,:],shiftT=self.CC.tps_start - self.T, s='PPSE_' + str(i))
        #
        #
        #     for i in range(5):
        #         self.mascene_LFPViewer.addline(self.pyrPPSI[i,index,:],shiftT=self.CC.tps_start - self.T, s='PPSI_' + str(i))
        #
        #     self.mascene_LFPViewer.addline(self.pyrPPSI_s[index, :], shiftT=self.CC.tps_start - self.T,s='pyrPPSI_s')
        #
        #     self.mascene_LFPViewer.addline(self.pyrPPSI_a[index, :], shiftT=self.CC.tps_start - self.T,s='pyrPPSI_a')
        #
        #
        #     print(self.CC.PreSynapticPos_AMPA[indices[index]], self.CC.PreSynapticPos_GABA[indices[index]],
        #           self.CC.PreSynaptic_Soma_GABA_d[indices[index]], \
        #           self.CC.PreSynaptic_Soma_GABA_s[indices[index]], self.CC.PreSynaptic_Soma_GABA_a[indices[index]])
        #     print(self.CC.Conx['PreSynapticPos_AMPA'][indices[index]],
        #           self.CC.Conx['PreSynapticPos_GABA'][indices[index]],
        #           self.CC.Conx['PreSynaptic_Soma_GABA_d'][indices[index]],
        #           self.CC.Conx['PreSynaptic_Soma_GABA_s'][indices[index]],
        #           self.CC.Conx['PreSynaptic_Soma_GABA_a'][indices[index]])
        #
        #     self.mascene_LFPViewer.clearfig()
        #     index = np.random.randint(0,self.pyrVs.shape[0])
        #     self.mascene_LFPViewer.addline(self.pyrVs[index, :], shiftT=self.CC.tps_start - self.T, s='pyrVs')
        #     self.mascene_LFPViewer.addline(self.pyrVd[index, :], shiftT=self.CC.tps_start - self.T, s='pyrVd')
        #     self.mascene_LFPViewer.addline(self.pyrVa[index, :], shiftT=self.CC.tps_start - self.T, s='pyrVa')
        #     index = np.random.randint(0,self.PV_Vs.shape[0])
        #     self.mascene_LFPViewer.addline(self.PV_Vs[index, :], shiftT=self.CC.tps_start - self.T, s='PV_Vs')
        #     index = np.random.randint(0,self.SST_Vs.shape[0])
        #     self.mascene_LFPViewer.addline(self.SST_Vs[index, :], shiftT=self.CC.tps_start - self.T, s='SST_Vs')
        #     index = np.random.randint(0,self.VIP_Vs.shape[0])
        #     self.mascene_LFPViewer.addline(self.VIP_Vs[index, :], shiftT=self.CC.tps_start - self.T, s='VIP_Vs')
        #     index = np.random.randint(0,self.RLN_Vs.shape[0])
        #     self.mascene_LFPViewer.addline(self.RLN_Vs[index, :], shiftT=self.CC.tps_start - self.T, s='RLN_Vs')
        #
        #
        #     self.mascene_LFPViewer.clearfig()
        #     index = np.random.randint(0,self.pyrVs.shape[0])
        #     # self.mascene_LFPViewer.addline(self.pyrVs[indices[index], :], shiftT=self.CC.tps_start - self.T, s='pyrVs')
        #     # self.mascene_LFPViewer.addline(self.pyrVd[indices[index], :], shiftT=self.CC.tps_start - self.T, s='pyrVd')
        #     # self.mascene_LFPViewer.addline(self.pyrVa[indices[index], :], shiftT=self.CC.tps_start - self.T, s='pyrVa')
        #     self.mascene_LFPViewer.addline(self.pyrPPSI_s[indices[index], :], shiftT=self.CC.tps_start - self.T,s='pyrPPSI_s')
        #     self.mascene_LFPViewer.addline(self.pyrPPSI_a[indices[index], :], shiftT=self.CC.tps_start - self.T,s='pyrPPSI_a')
        #     for i in range(5):
        #         self.mascene_LFPViewer.addline(self.pyrPPSE[i,index,:],shiftT=self.CC.tps_start - self.T, s='PPSE_' + str(i))
        #
        #
        #     for i in range(5):
        #         self.mascene_LFPViewer.addline(self.pyrPPSI[i,index,:],shiftT=self.CC.tps_start - self.T, s='PPSI_' + str(i))
        #
        #
        #
        #
        #     self.mascene_LFPViewer.clearfig()
        #     index = np.random.randint(0,1000)
        #     # self.mascene_LFPViewer.addline(np.sum(self.pyrPPSE[:,indices[index],:], axis=0),shiftT=self.CC.tps_start - self.T, s='PPSE_')
        #     # self.mascene_LFPViewer.addline(np.sum(self.pyrPPSI[:,indices[index],:], axis=0),shiftT=self.CC.tps_start - self.T, s='PPSI_')
        #     for i in range(5):
        #         self.mascene_LFPViewer.addline(self.pyrPPSE[i,indices[index],:],shiftT=self.CC.tps_start - self.T, s='PPSE_' + str(i))
        #     for i in range(5):
        #         self.mascene_LFPViewer.addline(self.pyrPPSI[i,indices[index],:],shiftT=self.CC.tps_start - self.T, s='PPSI_' + str(i))
        #     self.mascene_LFPViewer.addline(self.pyrPPSI_s[indices[index], :], shiftT=self.CC.tps_start - self.T,s='pyrPPSI_s')
        #     self.mascene_LFPViewer.addline(self.pyrPPSI_a[indices[index], :], shiftT=self.CC.tps_start - self.T,s='pyrPPSI_a')
        #     self.mascene_LFPViewer.addline(self.pyrPPSE_Dpyr[indices[index], :], shiftT=self.CC.tps_start - self.T,s='pyrdpyr')
        #     self.mascene_LFPViewer.addline(self.pyrPPSE_Th[indices[index], :], shiftT=self.CC.tps_start - self.T,s='pyrth')
        #


        self.flatindex = []
        for i in range(len(self.CellPosition)):
            for j in range(len(self.CellPosition[i])):
                self.flatindex.append([i, j])

        nb_pyr = 0
        cellspos = []
        List_cellsubtypes = []
        self.layers = []
        for i in range(len(self.flatindex)):
            l = self.flatindex[i][0]
            n = self.flatindex[i][1]
            if self.CC.List_celltypes[l][n] == 0:
                cellspos.append(self.CellPosition[l][n])
                List_cellsubtypes.append(self.CC.List_cellsubtypes[l][n])
                self.layers.append(l)
                nb_pyr += 1

        self.cellspos_PYR = np.array(cellspos)
        self.List_cellsubtypes_PYR = np.array(List_cellsubtypes)

        self.Color = []
        self.Sigs_dict = {}
        self.Sigs_dict["t"] = self.t #+ self.CC.tps_start
        self.CC.tps_start = self.t[-1]

        nb_pyr = 0
        nb_pv = 0
        nb_sst = 0
        nb_vip = 0
        nb_rln = 0
        count = 0
        for i in range(len(self.flatindex)):
            l = self.flatindex[i][0]
            n = self.flatindex[i][1]
            if self.List_Neurone_type[l][n] == 1:
                self.Color.append(self.List_Colors[l][n])
                self.Sigs_dict[self.List_Names[l][n]] = self.pyrVs[nb_pyr, :]
                nb_pyr += 1
            elif self.List_Neurone_type[l][n] == 2:
                self.Color.append(self.List_Colors[l][n])
                self.Sigs_dict[self.List_Names[l][n]] = self.PV_Vs[nb_pv, :]
                nb_pv += 1
            elif self.List_Neurone_type[l][n] == 3:
                self.Color.append(self.List_Colors[l][n])
                self.Sigs_dict[self.List_Names[l][n]] = self.SST_Vs[nb_sst, :]
                nb_sst += 1
            elif self.List_Neurone_type[l][n] == 4:
                self.Color.append(self.List_Colors[l][n])
                self.Sigs_dict[self.List_Names[l][n]] = self.VIP_Vs[nb_vip, :]
                nb_vip += 1
            elif self.List_Neurone_type[l][n] == 5:
                self.Color.append(self.List_Colors[l][n])
                self.Sigs_dict[self.List_Names[l][n]] = self.RLN_Vs[nb_rln, :]
                nb_rln += 1
            count += 1

        for i in range(self.DPYR_Vs.shape[0]):
            self.Color.append("#000000")
            self.Sigs_dict['DPYR' + str(i)] = self.DPYR_Vs[i, :]
            count += 1
        for i in range(self.Th_Vs.shape[0]):
            self.Color.append("#999999")
            self.Sigs_dict['Th' + str(i)] = self.Th_Vs[i, :]
            count += 1
        if self.displaycurves_CB.isChecked():
            print('update draw')
            # self.mascene_EEGViewer.setWindowSizeWithoutRedraw(int(self.T))
            self.mascene_EEGViewer.update(Sigs_dict=self.Sigs_dict, Colors=self.Color,
                                          percentage=float(self.displaycurve_per_e.text()))
            print('finish update draw')

        print(str(datetime.timedelta(seconds=int(time.time() - t0))))
        self.msg.close()
        self.parent.processEvents()

    def displaycurves_CB_fonc(self):
        if hasattr(self, 'Sigs_dict'):
            if self.displaycurves_CB.isChecked():
                # self.mascene_EEGViewer.setWindowSizeWithoutRedraw(int(self.T))
                self.mascene_EEGViewer.update(Sigs_dict=self.Sigs_dict, Colors=self.Color,
                                              percentage=float(self.displaycurve_per_e.text()))

    def displayVTK_CB_fonc(self):
        if self.displayVTK_CB.isChecked():
            try:
                self.Graph_viewer.draw_Graph()
                self.Graph_viewer.set_center()
                self.parent.processEvents()
            except:
                pass

    def Compute_LFP_fonc(self, meth=0,clear = None,s=''):

        if clear is None:
            if self.Compute_LFPClear_CB.isChecked():
                self.mascene_LFPViewer.clearfig()
                if s == '':
                    s = self.Compute_LFPtxt_LE.text()
        else:
            if clear:
                self.mascene_LFPViewer.clearfig()

        self.LFP = []
        self.LFP_Names = []
        print('Electrod disk')
        t0 = time.time()
        self.electrode_pos = [float(x.text()) for x in
                              [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]
        print(self.electrode_pos)
        self.electrode_disk = [1,
                               float(self.electrode_radius_e.text()),
                               float(self.electrode_angle1_e.text()),
                               float(self.electrode_angle2_e.text()),
                               float(self.electrode_cyl_h.text())]
        Fs = int(self.Fs_e.text())
        ComputeLFP = RecordedPotential.LFP(Fs=Fs,
                                           type=int(self.choose_electrode_type.currentText() == 'Cylinder'),
                                           re=self.electrode_disk[1], h=self.electrode_disk[4], tx=self.electrode_disk[2],
                                           ty=self.electrode_disk[3], pos=self.electrode_pos)

        if self.Compute_LFP2_type_CB.currentText() == "CSD" or self.Compute_LFPAll_CB.isChecked():
            self.save_d= {}
            self.save_d['Fs'] = Fs
            self.save_d['type'] = int(self.choose_electrode_type.currentText() == 'Cylinder')
            self.save_d['re'] = self.electrode_disk[1]
            self.save_d['h'] = self.electrode_disk[4]
            self.save_d['tx'] = self.electrode_disk[2]
            self.save_d['ty'] = self.electrode_disk[3]
            self.save_d['pos'] = self.electrode_posFs=Fs
            self.save_d['I_all'] = self.I_all
            self.save_d['PPS'] = self.PPS
            self.save_d['CellPosition'] = self.CellPosition
            self.save_d['List_celltypes'] = self.CC.List_celltypes
            self.save_d['List_cellsubtypes'] = self.CC.List_cellsubtypes
            self.save_d['Layertop_pos_mean'] = self.CC.Layertop_pos_mean

            LFP = ComputeLFP.compute_LFP_CSD(self.I_all, self.PPS,self.CellPosition, self.CC.List_celltypes,
                                         self.CC.List_cellsubtypes,self.CC.Layertop_pos_mean)
            # LFP = LFP / np.max(LFP)



            if self.type_tissu_Human.isChecked():
                Etype = 0
            else :
                Etype = 1

            Fs = int(1/(self.t[1]-self.t[0])) * 1000

            if self.choose_electrode_type.currentText() == 'Cylinder':
                cyl=1
            else:
                cyl=0

            e_g = Electrode.ElectrodeModel(re=self.electrode_disk[1], Etype=Etype, h=self.electrode_disk[4], cyl=cyl)
            LFP = -e_g.GetVelec(LFP, Fs=int(1000 / self.dt))

            self.LFP.append(LFP)
            self.LFP_Names.append('CSD'+ s )

            self.mascene_LFPViewer.addline(self.LFP[-1],shiftT=self.CC.tps_start - self.T, s='CSD' + s)

            self.forsave = {}
            self.forsave['List_cellsubtypes'] = self.CC.List_cellsubtypes
            self.forsave['List_Neurone_type'] = self.List_Neurone_type
            self.forsave['List_Names'] = self.List_Names
            self.forsave['Pyr_Pa'] = np.array([len(peakdet(p, 30)[0]) for p in self.pyrVd])
            self.forsave['PV_Pa'] = np.array([len(peakdet(p, 50)[0]) for p in self.PV_Vs])
            self.forsave['SST_Pa'] = np.array([len(peakdet(p, 50)[0]) for p in self.SST_Vs])
            self.forsave['VIP_Pa'] = np.array([len(peakdet(p, 50)[0]) for p in self.VIP_Vs])
            self.forsave['RLN_Pa'] = np.array([len(peakdet(p, 50)[0]) for p in self.RLN_Vs])
            self.forsave['W'] = self.W_plasticity[:,::250]
            self.forsave['LFP'] = LFP 
            self.forsave['Stim_EField'] = self.CC.Stim_EField

            # with open('FilePathName' + '_sigs.pkl', 'wb') as handler:
            #     pickle.dump(self.forsave, handler, protocol=pickle.HIGHEST_PROTOCOL)


    def UpdateLFP(self):
        if self.Temporal_PSD_CB.isChecked():
            self.mascene_LFPViewer.updatePSD(shiftT=self.CC.tps_start - self.T)
        else:
            self.mascene_LFPViewer.update(shiftT=self.CC.tps_start - self.T)

    def Compute_LFPDiskCoated_fonc(self):
        self.electrode_pos = [float(x.text()) for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]
        self.electrode_disk = [float(self.electrode_radius_e.text()),
                               float(self.electrode_angle1_e.text()),
                               float(self.electrode_angle2_e.text())]

        ind = self.Compute_LFPDiskCoated_type_CB.currentIndex()
        # if ind in [0, 1, 2]:
        #     coated = 0
        # else:
        #     coated = 1
        coated = self.Compute_LFPDisk_coated_CB.isChecked()
        e_g = Electrode.ElectrodeModel(re=self.electrode_disk[0], Etype=ind, coated=coated)
        LFP = e_g.GetVelec(self.LFP, Fs=int(1000 / self.dt))
        self.mascene_LFPViewer.addLFP(LFP, shiftT=self.CC.tps_start - self.T)

    def SaveSimul(self,Filepath = None):
        extension = "txt"
        if not isinstance(Filepath, str):
            fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=extension + " (*." + extension + ")")
            if (fileName[0] == ''):
                return
            if os.path.splitext(fileName[0])[1] == '':
                fileName = (fileName[0] + '.' + extension, fileName[1])
        else:
            fileName = [Filepath,'txt']
        if fileName[1] in extension + " (*." + extension + ")":
            f = open(fileName[0], 'w')
            if self.type_tissu_Human.isChecked():
                f.write("TypeTissu" + "\t" + "Human\n")
            elif self.type_tissu_Rat.isChecked():
                f.write("TypeTissu" + "\t" + "Rat\n")
            elif self.type_tissu_Mouse.isChecked():
                f.write("TypeTissu" + "\t" + "Mouse\n")

            f.write("D" + "\t" + str(self.CC.D) + "\n")
            f.write("L" + "\t" + str(self.CC.L) + "\n")
            f.write("Layer_d" + "\t" + str(self.CC.Layer_d.tolist()) + "\n")

            f.write("Layer_nbCells" + "\t" + str(self.CC.Layer_nbCells.tolist()) + "\n")
            f.write("PYRpercent" + "\t" + str(self.CC.C.PYRpercent.tolist()) + "\n")
            f.write("PVpercent" + "\t" + str(self.CC.C.PVpercent.tolist()) + "\n")
            f.write("SSTpercent" + "\t" + str(self.CC.C.SSTpercent.tolist()) + "\n")
            f.write("VIPpercent" + "\t" + str(self.CC.C.VIPpercent.tolist()) + "\n")
            f.write("RLNpercent" + "\t" + str(self.CC.C.RLNpercent.tolist()) + "\n")

            f.write("PCsubtypes_Per" + "\t" + str(self.CC.PCsubtypes_Per.tolist()) + "\n")
            f.write("NB_PYR" + "\t" + str(self.CC.C.NB_PYR.tolist()) + "\n")
            f.write("NB_IN" + "\t" + str(self.CC.C.NB_IN.tolist()) + "\n")
            f.write("NB_PV" + "\t" + str(self.CC.C.NB_PV.tolist()) + "\n")
            f.write("NB_PV_BC" + "\t" + str(self.CC.C.NB_PV_BC.tolist()) + "\n")
            f.write("NB_PV_ChC" + "\t" + str(self.CC.C.NB_PV_ChC.tolist()) + "\n")
            f.write("NB_SST" + "\t" + str(self.CC.C.NB_SST.tolist()) + "\n")
            f.write("NB_VIP" + "\t" + str(self.CC.C.NB_VIP.tolist()) + "\n")
            f.write("NB_RLN" + "\t" + str(self.CC.C.NB_RLN.tolist()) + "\n")

            f.write("List_Lambda_s" + "\t" + str(self.CC.C.List_Lambda_s.tolist()) + "\n")
            f.write("List_Lambda_d" + "\t" + str(self.CC.C.List_Lambda_d.tolist()) + "\n")
            f.write("List_Lambda_PV" + "\t" + str(self.CC.C.List_Lambda_PV.tolist()) + "\n")
            f.write("List_Lambda_SST" + "\t" + str(self.CC.C.List_Lambda_SST.tolist()) + "\n")
            f.write("List_Lambda_VIP" + "\t" + str(self.CC.C.List_Lambda_VIP.tolist()) + "\n")
            f.write("List_Lambda_RLN" + "\t" + str(self.CC.C.List_Lambda_RLN.tolist()) + "\n")

            f.write("PCsubtypes_Percentage" + "\t" + str(self.CC.C.PCsubtypes_Percentage.tolist()) + "\n")

            f.write("Afference_type" + "\t" + str(self.r0.isChecked()) + "\n")
            f.write("Afferences" + "\t" + str(self.CC.Afferences.tolist()) + "\n")

            f.write("StimDuration" + "\t" + self.StimDuration_e.text() + "\n")
            f.write("i_inj" + "\t" + self.i_inj_e.text() + "\n")
            f.write("tau" + "\t" + self.tau_e.text() + "\n")
            f.write("nbStim" + "\t" + self.nbStim_e.text() + "\n")
            f.write("varianceStim" + "\t" + self.varianceStim_e.text() + "\n")
            f.write("seed" + "\t" + self.seed_e.text() + "\n")

            f.write("TH_StimDuration_e" + "\t" + self.TH_StimDuration_e.text() + "\n")
            f.write("TH_i_inj_e" + "\t" + self.TH_i_inj_e.text() + "\n")
            f.write("TH_tau_e" + "\t" + self.TH_tau_e.text() + "\n")
            f.write("TH_nbStim_e" + "\t" + self.TH_nbStim_e.text() + "\n")
            f.write("TH_deltamin_e" + "\t" + self.TH_deltamin_e.text() + "\n")
            f.write("TH_delta_e" + "\t" + self.TH_delta_e.text() + "\n")

            f.write("SimDuration" + "\t" + self.SimDuration_e.text() + "\n" )
            f.write("Fs" + "\t" + self.Fs_e.text() + "\n")
            f.write("Type" + "\t" + self.Stimtype_CB.currentText() + "\n")
            f.write("Periode" + "\t" + self.StimPeriode_e.text() + "\n")
            f.write("Stimstart" + "\t" + self.StimStart_e.text() + "\n")

            f.write("Cellpos1" + "\t" + str(self.CellPosition[0].tolist()) + "\n")
            f.write("Cellpos23" + "\t" + str(self.CellPosition[1].tolist()) + "\n")
            f.write("Cellpos4" + "\t" + str(self.CellPosition[2].tolist()) + "\n")
            f.write("Cellpos5" + "\t" + str(self.CellPosition[3].tolist()) + "\n")
            f.write("Cellpos6" + "\t" + str(self.CellPosition[4].tolist()) + "\n")

            f.write("cell_placement_CB" + "\t" + self.cell_placement_CB.currentText() + "\n")
            f.write("cellplace" + "\t" + self.seed_place.text() + "\n")

            f.write("electrode_x" + "\t" + self.electrode_x_e.text() + "\n")
            f.write("electrode_y" + "\t" + self.electrode_y_e.text() + "\n")
            f.write("electrode_z" + "\t" + self.electrode_z_e.text() + "\n")

            f.write("List_Names" + "\t" + str(self.List_Names) + "\n")
            f.write("List_Colors" + "\t" + str(self.List_Colors) + "\n")
            f.write("List_Neurone_type" + "\t" + str(self.List_Neurone_type) + "\n")

            for dictlayer in self.CC.List_Neurone_param:
                for dict_param in dictlayer:
                    f.write("dict_param" + "\t" + str(dict_param) + "\n")

            f.close()
        # except:
        #     msg_cri('Not able to save the simulation')

    def LoadSimul(self,Filepath=None):
        extension = "txt"
        if not isinstance(Filepath, str):
            fileName = QFileDialog.getOpenFileName(caption='Load parameters', filter=extension + " (*." + extension + ")")
            if (fileName[0] == ''):
                return
            if os.path.splitext(fileName[0])[1] == '':
                fileName = (fileName[0] + '.' + extension, fileName[1])
        else:
            fileName = [Filepath,'txt']
        if fileName[1] in extension + " (*." + extension + ")":

            with open(fileName[0], 'r') as f:
                line = f.readline()
                type = line.split("\t")[-1].replace('\n', '').replace(" ", "")
                if type == "Human":
                    self.type_tissu_Human.setChecked(True)
                elif type == "Rat":
                    self.type_tissu_Rat.setChecked(True)
                elif type == "Mouse":
                    self.type_tissu_Mouse.setChecked(True)

                line = f.readline()
                self.D_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.L_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.Layer_d = [float(l) for l in
                                line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                  "").split(" ")]

                self.Layer_d1_l.setText(str(self.Layer_d[0]))
                self.Layer_d23_l.setText(str(self.Layer_d[1]))
                self.Layer_d4_l.setText(str(self.Layer_d[2]))
                self.Layer_d5_l.setText(str(self.Layer_d[3]))
                self.Layer_d6_l.setText(str(self.Layer_d[4]))
                self.set_tissue_func()

                line = f.readline()
                Layer_nbCells = [int(l) for l in
                                 line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                  "").split(
                                     " ")]
                line = f.readline()
                PYRpercent = [float(l) for l in
                              line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                               "").split(
                                  " ")]
                line = f.readline()
                PVpercent = [float(l) for l in
                             line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                              "").split(
                                 " ")]
                line = f.readline()
                SSTpercent = [float(l) for l in
                              line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                               "").split(
                                  " ")]
                line = f.readline()
                VIPpercent = [float(l) for l in
                              line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                               "").split(
                                  " ")]
                line = f.readline()
                RLNpercent = [float(l) for l in
                              line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                               "").split(
                                  " ")]

                line = f.readline()
                PCsubtypes_Per = np.array([[float(r) for r in l.split(", ")] for l in
                                           line.split("\t")[-1].replace('\n', '').replace('[[', '').replace(']]', '').split(
                                               "], [")])
                line = f.readline()
                NB_PYR = np.array([int(l) for l in
                                   line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                    "").split(
                                       " ")])
                line = f.readline()
                NB_IN = np.array([int(l) for l in
                                  line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                   "").split(
                                      " ")])
                line = f.readline()
                NB_PV = np.array([int(l) for l in
                                  line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                   "").split(
                                      " ")])
                line = f.readline()
                NB_PV_BC = np.array([int(l) for l in
                                     line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                      "").split(
                                         " ")])
                line = f.readline()
                NB_PV_ChC = np.array([int(l) for l in
                                      line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                       "").split(
                                          " ")])
                line = f.readline()
                NB_SST = np.array([int(l) for l in
                                   line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                    "").split(
                                       " ")])
                line = f.readline()
                NB_VIP = np.array([int(l) for l in
                                   line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                    "").split(
                                       " ")])
                line = f.readline()
                NB_RLN = np.array([int(l) for l in
                                   line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                    "").split(
                                       " ")])

                self.CC.update_cellNumber(np.array(Layer_nbCells),
                                          np.array(PYRpercent),
                                          np.array(PVpercent),
                                          np.array(SSTpercent),
                                          np.array(VIPpercent),
                                          np.array(RLNpercent),
                                          PCsubtypes_Per=np.array(PCsubtypes_Per),
                                          NB_PYR=NB_PYR,
                                          NB_PV_BC=NB_PV_BC,
                                          NB_PV_ChC=NB_PV_ChC,
                                          NB_IN=NB_IN,
                                          NB_PV=NB_PV,
                                          NB_SST=NB_SST,
                                          NB_VIP=NB_VIP,
                                          NB_RLN=NB_RLN
                                          )

                self.nbcellsnb1.setText(str(int(self.CC.Layer_nbCells[0])))
                self.nbcellsnb23.setText(str(int(self.CC.Layer_nbCells[1])))
                self.nbcellsnb4.setText(str(int(self.CC.Layer_nbCells[2])))
                self.nbcellsnb5.setText(str(int(self.CC.Layer_nbCells[3])))
                self.nbcellsnb6.setText(str(int(self.CC.Layer_nbCells[4])))
                self.nbcellsnbtotal.setText(str(int(np.sum(self.CC.Layer_nbCells))))

                # self.update_cellNnumber()


                for i, l in enumerate(self.List_PYRpercent):
                    l.setText(str(PYRpercent[i]))

                for i, l in enumerate(self.List_PVpercent):
                    l.setText(str(PVpercent[i]))

                for i, l in enumerate(self.List_SSTpercent):
                    l.setText(str(SSTpercent[i]))

                for i, l in enumerate(self.List_VIPpercent):
                    l.setText(str(VIPpercent[i]))

                for i, l in enumerate(self.List_RLNpercent):
                    l.setText(str(RLNpercent[i]))


                line = f.readline()
                List_Lambda_s = np.array([[float(r) for r in l.split(", ")] for l in
                                          line.split("\t")[-1].replace('\n', '').replace('[[', '').replace(']]',
                                                                                                           '').split(
                                              "], [")])
                i = 0
                for l in range(List_Lambda_s.shape[0]):
                    for c in range(List_Lambda_s.shape[1]):
                        self.List_Lambda_S[i].setText(str(List_Lambda_s[l, c]))
                        i += 1

                        if l+1 == 1 and not self.CC.List_Lambda_PC23_s[c][0] ==0:
                            self.CC.List_Lambda_PC23_s[c] = (self.CC.List_Lambda_PC23_s[c]/ self.CC.List_Lambda_PC23_s[c][0]) * List_Lambda_s[l, c]
                        elif l+1 == 2 and not self.CC.List_Lambda_PC4_s[c][0] ==0:
                            self.CC.List_Lambda_PC4_s[c] = (self.CC.List_Lambda_PC4_s[c]/ self.CC.List_Lambda_PC4_s[c][0]) * List_Lambda_s[l, c]
                        elif l + 1 == 3 and not self.CC.List_Lambda_PC5_s[c][0] ==0:
                            self.CC.List_Lambda_PC5_s[c] = (self.CC.List_Lambda_PC5_s[c]/ self.CC.List_Lambda_PC5_s[c][0]) * List_Lambda_s[l, c]
                        elif l + 1 == 4 and not self.CC.List_Lambda_PC6_s[c][0] ==0:
                            self.CC.List_Lambda_PC6_s[c] = (self.CC.List_Lambda_PC6_s[c]/ self.CC.List_Lambda_PC6_s[c][0]) * List_Lambda_s[l, c]





                line = f.readline()
                List_Lambda_d = np.array([[float(r) for r in l.split(", ")] for l in
                                          line.split("\t")[-1].replace('\n', '').replace('[[', '').replace(']]', '').split(
                                              "], [")])
                i=0
                for l in range(List_Lambda_d.shape[0]):
                    for c in range(List_Lambda_d.shape[1]):
                        self.List_Lambda_D[i].setText(str(List_Lambda_d[l,c]))
                        i += 1

                        if l + 1 == 1 and not self.CC.List_Lambda_PC23_d[c][0] == 0:
                            self.CC.List_Lambda_PC23_d[c] = (self.CC.List_Lambda_PC23_d[c] / self.CC.List_Lambda_PC23_d[c][0]) * List_Lambda_d[ l, c]
                        elif l + 1 == 2 and not self.CC.List_Lambda_PC4_d[c][0] == 0:
                            self.CC.List_Lambda_PC4_d[c] = (self.CC.List_Lambda_PC4_d[c] /
                                                            self.CC.List_Lambda_PC4_d[c][0]) * List_Lambda_d[
                                                               l, c]
                        elif l + 1 == 3 and not self.CC.List_Lambda_PC5_d[c][0] == 0:
                            self.CC.List_Lambda_PC5_d[c] = (self.CC.List_Lambda_PC5_d[c] /
                                                            self.CC.List_Lambda_PC5_d[c][0]) * List_Lambda_d[
                                                               l, c]
                        elif l + 1 == 4 and not self.CC.List_Lambda_PC6_d[c][0] == 0:
                            self.CC.List_Lambda_PC6_d[c] = (self.CC.List_Lambda_PC6_d[c] /
                                                            self.CC.List_Lambda_PC6_d[c][0]) * List_Lambda_d[
                                                                       l, c]


                line = f.readline()
                List_Lambda_PV = np.array([float(l) for l in
                                line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",","").split(" ")])


                line = f.readline()
                List_Lambda_SST = np.array([float(l) for l in
                                line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                  "").split(" ")])
                line = f.readline()
                List_Lambda_VIP = np.array([float(l) for l in
                                line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                  "").split(" ")])
                line = f.readline()
                List_Lambda_RLN = np.array([float(l) for l in
                                line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",
                                                                                                                  "").split(" ")])

                i = 0
                for l in range(List_Lambda_PV.shape[0]):
                        self.List_Lambda_IN[i].setText(str(List_Lambda_PV[l]))
                        i += 1
                        self.List_Lambda_IN[i].setText(str(List_Lambda_SST[l]))
                        i += 1
                        self.List_Lambda_IN[i].setText(str(List_Lambda_VIP[l]))
                        i += 1
                        self.List_Lambda_IN[i].setText(str(List_Lambda_RLN[l]))
                        i += 1



                for c in range(List_Lambda_PV.shape[0]):
                    if self.CC.List_Lambda_PV[c+1][0] == 0:
                        self.CC.List_Lambda_PV[c+1] = (self.CC.List_Lambda_PV[c+1] / self.CC.List_Lambda_PV[c+1][0]) * \
                                                        List_Lambda_PV[c]
                for c in range(List_Lambda_SST.shape[0]):
                    if self.CC.List_Lambda_SST[c+1][0] == 0:
                        self.CC.List_Lambda_SST[c+1] = (self.CC.List_Lambda_SST[c+1] / self.CC.List_Lambda_SST[c+1][0]) * \
                                                        List_Lambda_SST[c]
                for c in range(List_Lambda_VIP.shape[0]):
                    if self.CC.List_Lambda_VIP[c+1][0] == 0:
                        self.CC.List_Lambda_VIP[c+1] = (self.CC.List_Lambda_VIP[c+1] / self.CC.List_Lambda_VIP[c+1][0]) * \
                                                        List_Lambda_VIP[c]
                for c in range(List_Lambda_RLN.shape[0]):
                    if self.CC.List_Lambda_RLN[c+1][0] == 0:
                        self.CC.List_Lambda_RLN[c+1] = (self.CC.List_Lambda_RLN[c+1] / self.CC.List_Lambda_RLN[c+1][0]) * \
                                                        List_Lambda_RLN[c]

                self.update_lambda()

                line = f.readline()
                PCsubtypes_Percentage = np.array([[float(r) for r in l.split(", ")] for l in
                                          line.split("\t")[-1].replace('\n', '').replace('[[', '').replace(']]', '').split(
                                              "], [")])
                i=0
                for l in range(PCsubtypes_Percentage.shape[0]):
                    for c in range(PCsubtypes_Percentage.shape[1]):
                        self.List_PCsubtypes[i].setText(str(PCsubtypes_Percentage[l,c]))
                        i += 1

                # self.CC.C.update_morphology( NB_PYR = NB_PYR,
                #             NB_PV_BC = NB_PV_BC,
                #             NB_PV_ChC = NB_PV_ChC,
                #             NB_IN = NB_IN,
                #             NB_PV = NB_PV,
                #             NB_SST = NB_SST,
                #             NB_VIP = NB_VIP,
                #             NB_RLN = NB_RLN)

                line = f.readline()
                Afference_type = line.split("\t")[-1].replace('\n', '').replace(" ", "")
                if Afference_type == 'True':
                    self.r0.setChecked(True)
                else:
                    self.r1.setChecked(True)

                line = f.readline()
                Afferences = np.array([[float(r) for r in l.split(", ")] for l in
                                       line.split("\t")[-1].replace('\n', '').replace('[[', '').replace(']]', '').split(
                                           "], [")])
                self.CC.update_connections(Afferences, fixed=not self.r0.isChecked())

                line = f.readline()
                self.StimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.i_inj_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.tau_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.nbStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.varianceStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.seed_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

                line = f.readline()
                self.TH_StimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.TH_i_inj_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.TH_tau_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.TH_nbStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.TH_deltamin_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.TH_delta_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

                line = f.readline()
                self.SimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.Fs_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

                line = f.readline()
                index =  self.Stimtype_CB.findText(line.split("\t")[-1].replace('\n', ''))
                self.Stimtype_CB.setCurrentIndex(index)
                line = f.readline()
                self.StimPeriode_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.StimStart_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

                line = f.readline()
                sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split(
                    '],[')
                Cellpos = []
                for s in sublist:
                    Cellpos.append([float(s2) for s2 in s.split(',')])
                self.CellPosition = [np.array(Cellpos)]
                line = f.readline()
                sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split(
                    '],[')
                Cellpos = []
                for s in sublist:
                    Cellpos.append([float(s2) for s2 in s.split(',')])
                self.CellPosition.append(np.array(Cellpos))
                line = f.readline()
                sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split(
                    '],[')
                Cellpos = []
                for s in sublist:
                    Cellpos.append([float(s2) for s2 in s.split(',')])
                self.CellPosition.append(np.array(Cellpos))
                line = f.readline()
                sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split(
                    '],[')
                Cellpos = []
                for s in sublist:
                    Cellpos.append([float(s2) for s2 in s.split(',')])
                self.CellPosition.append(np.array(Cellpos))
                line = f.readline()
                sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split(
                    '],[')
                Cellpos = []
                for s in sublist:
                    Cellpos.append([float(s2) for s2 in s.split(',')])
                self.CellPosition.append(np.array(Cellpos))
                # self.CellPosition = self.CC.Cellpos

                line = f.readline()
                index = self.cell_placement_CB.findText(line.split("\t")[-1].replace('\n', ''),
                                                        Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive)
                self.cell_placement_CB.setCurrentIndex(index)

                line = f.readline()
                seed = line.split("\t")[-1].replace('\n', '').replace(" ", "")
                self.seed_place.setText(seed)
                seed = int(seed)
                print(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR, self.CC.NB_Th, self.CellPosition,seed)
                self.CC.Conx = Connectivity.Create_Connectivity_Matrix(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR,
                                                                       self.CC.NB_Th, np.array(self.CellPosition,dtype='object'), seed=seed)


                line = f.readline()
                self.electrode_x_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.electrode_y_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                self.electrode_z_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
                line = f.readline()
                List_Names = line.split("\t")[-1].replace('\n', '').replace('[', '').replace(']', '').replace("'",
                                                                                                              '').replace(
                    " ", '').split(',')
                index = 0
                self.List_Names = []
                for l in range(len(self.CellPosition)):
                    length = len(self.CellPosition[l])
                    self.List_Names.append(List_Names[index:index + length])
                    index += length
                line = f.readline()
                List_Colors = line.split("\t")[-1].replace('\n', '').replace('[', '').replace(']', '').replace("'",
                                                                                                               '').replace(
                    " ", '').split(',')
                index = 0
                self.List_Colors = []
                for l in range(len(self.CellPosition)):
                    length = len(self.CellPosition[l])
                    self.List_Colors.append(List_Colors[index:index + length])
                    index += length
                line = f.readline()
                List_Neurone_type = [int(l) for l in
                                     line.split("\t")[-1].replace('\n', '').replace('[', '').replace(']', '').replace("'",
                                                                                                                      '').replace(
                                         " ", '').split(',')]
                index = 0
                self.List_Neurone_type = []
                for l in range(len(self.CellPosition)):
                    length = len(self.CellPosition[l])
                    self.List_Neurone_type.append(List_Neurone_type[index:index + length])
                    index += length
                List_Neurone_param = []
                for line in f:
                    finalstring = line.split("\t")[-1].replace('\n', '').replace("{", "").replace("}", "").replace("'",
                                                                                                                   "").replace(
                        " ", "").replace(" ", "")

                    # Splitting the string based on , we get key value pairs
                    listdict = finalstring.split(",")

                    dictionary = {}
                    for i in listdict:
                        # Get Key Value pairs separately to store in dictionary
                        keyvalue = i.split(":")

                        # Replacing the single quotes in the leading.
                        m = keyvalue[0].strip('\'')
                        m = m.replace("\"", "")
                        dictionary[m] = float(keyvalue[1].strip('"\''))
                    List_Neurone_param.append(dictionary)

                index = 0
                List_Neurone_param2 = []
                for l in range(len(self.CellPosition)):
                    length = len(self.CellPosition[l])
                    List_Neurone_param2.append(List_Neurone_param[index:index + length])
                    index += length
                self.createCells = True
                self.update_Model()
                self.CC.List_Neurone_param = List_Neurone_param2

                self.electrode_placement_func()
                self.ApplyModXNMM( List_Neurone_param2, self.List_Names, self.List_Colors)
                self.masceneCM.update()
                # self.update_graph()

    def SaveRes(self):

        Sigs = {}
        Sigs['PPS'] = self.PPS
        Sigs['Epos'] = self.electrode_pos
        Sigs['CellPos'] = self.CellPosition
        Sigs['list'] = self.CC.List_celltypes
        Sigs['sublist'] =self.CC.List_cellsubtypes
        Sigs['layers'] = self.layers
        Sigs['layertop'] = self.CC.Layertop_pos_mean
        Sigs['Vs'] = self.pyrVs
        Sigs['Vs_Dpyr'] = self.DPYR_Vs
        Sigs['Vs_Th'] = self.Th_Vs
        Sigs['Vs_PV'] = self.PV_Vs
        Sigs['Vs_SST'] = self.SST_Vs
        Sigs['Vs_VIP'] = self.VIP_Vs
        Sigs['Vs_RLN'] = self.RLN_Vs
        Sigs['t'] = self.t
        Sigs['I_all'] = self.I_all


        scipy.io.savemat('Sigs', mdict=Sigs)


        if hasattr(self, 'Sigs_dict'):
            exPopup = QuestionWhatToSave(self)
            if exPopup.exec() == QDialog.DialogCode.Accepted:
                saveLFP = exPopup.save_lfp_CB.isChecked()
                savesignals = exPopup.save_signal_CB.isChecked()
                savecoords = exPopup.save_coord_CB.isChecked()
                if not saveLFP and not savesignals and not savecoords:
                    return
            else:
                return
            exPopup.deleteLater()
            if savesignals:
                Sigs_dict = {'PPS':self.PPS,
                             'CellPosition': self.CellPosition,
                             'celltypes':self.CC.List_celltypes,
                             'cellsubtypes':self.CC.List_cellsubtypes,
                             'Layertop_pos_mean':self.CC.Layertop_pos_mean,
                             'layers':self.layers,
                              'Fs':self.Fs_e,
                             't':self.t}

            else:
                LFPs, Names, Fs = self.mascene_EEGViewer.Get_Sig_as_array( )
                Sigs_dict = {'LFP':LFPs,
                             'Names':Names,
                             'Fs':Fs}

            if not 't' in Sigs_dict.keys():
                Sigs_dict["t"] = self.t

            if hasattr(self, 'LFP') and saveLFP:
                for i in range(len(self.LFP )):
                    Sigs_dict['LFP_'+self.LFP_Names[i]] = self.LFP[i]

            if hasattr(self, 'CellPosition') and savecoords:
                CellPosition= []
                for i in self.flatindex:
                    CellPosition.append(self.CellPosition[i[0]][i[1]])
                Sigs_dict['Coordinates'] = CellPosition

            if savecoords:
                fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=".mat (*.mat)")
            else:
                fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=".mat (*.mat)")

            if (fileName[0] == ''):
                return
            tp = Sigs_dict['t']
            if fileName[1] == '.data (*.data)':
                file_pi = open(fileName[0], 'wb')
                pickle.dump(Sigs_dict, file_pi, -1)
                file_pi.close()
            elif fileName[1] == '.bin (*.bin)':
                # write .des file
                path, name = os.path.split(fileName[0])
                name = name.split('.')[0]
                file = open(os.path.join(path, name + '.des'), "w")
                file.write("[patient]  x" + '\n')
                file.write("[date] " + datetime.datetime.today().strftime('%m/%d/%Y') + '\n')
                file.write("[time] " + datetime.datetime.today().strftime('%H:%M:%S') + '\n')
                Fs = int(1. / (tp[1] - tp[0]))
                file.write("[samplingfreq] " + str(Fs) + '\n')
                file.write("[nbsegments] 1" + '\n')
                file.write("[enabled] 1" + '\n')
                nsample = len(tp)
                file.write("[nbsamples] " + str(nsample) + '\n')
                file.write("[segmentsize] " + str(nsample) + '\n')
                file.write("[segmentInitialTimes] 0.0" + '\n')
                file.write("[nbchannels] " + str(len(Sigs_dict)) + '\n')
                file.write("[channelnames] :" + '\n')
                for s in Sigs_dict.keys():
                    file.write(s + " ------" + '\n')
                # file.write('aaa'+" ------"+ '\n')
                file.close()
                keys = list(Sigs_dict.keys())
                array = np.array(Sigs_dict[keys[0]])
                for s in keys[1:]:
                    array = np.vstack((array, Sigs_dict[s] * 1000))
                array = array.T.flatten()
                array.astype('float32')
                s = struct.pack('f' * len(array), *array)
                file = open(os.path.join(path, name + '.bin'), "wb")
                # array.tofile(file)
                file.write(s)
                file.close()

            elif fileName[1] == '.mat (*.mat)':
                scipy.io.savemat(fileName[0], mdict=Sigs_dict)
            elif fileName[1] == '.csv (*.csv)':
                f = open(fileName[0], 'w')
                w = csv.writer(f, delimiter='\t', lineterminator='\n')
                w.writerow(Sigs_dict.keys())
                for values in Sigs_dict.values():
                    w.writerow(['{:e}'.format(var) for var in values])
                f.close()
            elif fileName[1] == '.dat (*.dat)':
                # write .des file
                path, name = os.path.split(fileName[0])
                name = name.split('.')[0]
                file = open(os.path.join(path, name + '.des'), "w")
                file.write("[patient]  x" + '\n')
                file.write("[date] " + datetime.datetime.today().strftime('%m/%d/%Y') + '\n')
                file.write("[time] " + datetime.datetime.today().strftime('%H:%M:%S') + '\n')
                Fs = int(1. / (tp[1] - tp[0]))
                file.write("[samplingfreq] " + str(Fs) + '\n')
                file.write("[nbsegments] 1" + '\n')
                file.write("[enabled] 1" + '\n')
                nsample = len(tp)
                file.write("[nbsamples] " + str(nsample) + '\n')
                file.write("[segmentsize] " + str(nsample) + '\n')
                file.write("[segmentInitialTimes] 0.0" + '\n')
                file.write("[nbchannels] " + str(len(Sigs_dict)) + '\n')
                file.write("[channelnames] :" + '\n')
                for s in Sigs_dict.keys():
                    file.write(s + " ------" + '\n')
                #
                file.close()
                f = open(fileName[0], 'w')
                w = csv.writer(f, delimiter=' ', lineterminator='\n')
                for idx in range(len(tp)):
                    line = []
                    for i_v, values in enumerate(Sigs_dict.values()):
                        if i_v == 0:
                            line.append(values[idx])
                        else:
                            line.append(values[idx] * 1000)
                    w.writerow(['{:f}'.format(var) for var in line])
                f.close()
            elif fileName[1] == '.edf (*.edf)':
                Fs = int(1. / (tp[1] - tp[0]))
                Sigs_dict.pop("t")
                N = len(Sigs_dict.keys())
                f = pyedflib.EdfWriter(fileName[0], N, file_type=1)

                lims = 0
                for key, value in Sigs_dict.items():
                    if key not in 't':
                        lims = max(lims, np.max(np.abs(value)))

                lims *= 2
                for i, key in enumerate(Sigs_dict.keys()):
                    f.setLabel(i, key)
                    f.setSamplefrequency(i, Fs)
                    f.setPhysicalMaximum(i, lims)
                    f.setPhysicalMinimum(i, -lims)
                f.update_header()

                kiter = len(tp) // Fs
                for k in range(kiter):
                    for i, key in enumerate(Sigs_dict.keys()):
                        f.writePhysicalSamples(Sigs_dict[key][k * Fs:(k + 1) * Fs].flatten())

                f.update_header()
                f.close()

        return

    def LoadModel(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Model", "", "Cortical column (*.py)")
        if fileName[0] == '':
            return
        if fileName[1] == "Cortical column (*.py)":
            if os.path.splitext(fileName[0])[1] == '':
                fileName[0] = (fileName[0] + '.py', fileName[1])
            fileName = str(fileName[0])
            (filepath, filename) = os.path.split(fileName)
            sys.path.append(filepath)
            (shortname, extension) = os.path.splitext(filename)
            self.Colonne_class = __import__(shortname)
            self.Colonne = getattr(self.Colonne_class, 'Colonne_cortical')
            self.CC = self.Colonne()
            self.CC.updateTime.something_happened.connect(self.updateTime)
            self.createCells = True

    def Change_Lambda_type(self):
        if self.ld0.isChecked():
            self.CC.var_lambdas = 0
            # List_Lambda_s = self.CC.C.List_Lambda_s
            # List_Lambda_s.reshape(0,1)
            #
            # for l in range(List_Lambda_s.shape[0]):
            #         self.List_Lambda_S[l].setText(str(List_Lambda_s[l]))
            i=0
            for l in range(self.CC.C.List_Lambda_s.shape[0]):
                for c in range(self.CC.C.List_Lambda_s.shape[1]):
                    self.List_Lambda_S[i].setText(str(self.CC.C.List_Lambda_s[l,c]))
                    i += 1
            i=0
            for l in range(self.CC.C.List_Lambda_d.shape[0]):
                for c in range(self.CC.C.List_Lambda_d.shape[1]):
                    self.List_Lambda_D[i].setText(str(self.CC.C.List_Lambda_d[l, c]))
                    i += 1

            i = 0
            for l in range(len(self.CC.C.List_Lambda_PV)):
                self.List_Lambda_IN[i].setText(str(self.CC.C.List_Lambda_PV[l]))
                i += 1
                self.List_Lambda_IN[i].setText(str(self.CC.C.List_Lambda_SST[l]))
                i += 1
                self.List_Lambda_IN[i].setText(str(self.CC.C.List_Lambda_VIP[l]))
                i += 1
                self.List_Lambda_IN[i].setText(str(self.CC.C.List_Lambda_RLN[l]))
                i += 1




        elif self.ld1.isChecked():
            self.CC.var_lambdas=1
            List_Lambda_PC23 = self.CC.C.List_Lambda_PC23
            List_Lambda_PC4 = self.CC.C.List_Lambda_PC4
            List_Lambda_PC5 = self.CC.C.List_Lambda_PC5
            List_Lambda_PC6 = self.CC.C.List_Lambda_PC6

            List_Lambda_PV_f = self.CC.C.List_Lambda_PV_f
            List_Lambda_SST_f = self.CC.C.List_Lambda_SST_f
            List_Lambda_VIP_f = self.CC.C.List_Lambda_VIP_f
            List_Lambda_RLN_f = self.CC.C.List_Lambda_RLN_f
            for l in range(len(self.List_Lambda_D)):
                self.List_Lambda_D[l].setText('0')


            for l in range(4):
                self.List_Lambda_S[l].setText(str(List_Lambda_PC23[0]))
                self.List_Lambda_S[l+5].setText(str(List_Lambda_PC4[0]))
                self.List_Lambda_S[l+10].setText(str(List_Lambda_PC5[0]))
                self.List_Lambda_S[l+15].setText(str(List_Lambda_PC6[0]))
            for l in range(4):
                self.List_Lambda_S[5 * l + 2].setText('0')
                self.List_Lambda_S[5 * l + 3].setText('0')
                self.List_Lambda_S[5*l+4].setText('0')

            for l in range(4):
                self.List_Lambda_IN[4*l].setText(str(List_Lambda_PV_f[0]))
                self.List_Lambda_IN[4*l+1].setText(str(List_Lambda_SST_f[0]))
                self.List_Lambda_IN[4*l+2].setText(str(List_Lambda_VIP_f[0]))
                self.List_Lambda_IN[4*l+3].setText(str(List_Lambda_RLN_f[0]))



    def ChangeModel_type(self ):
        if self.type_tissu_Human.isChecked():
            type = 0 #: 'human, 1:Rat, 0:mice
        elif self.type_tissu_Rat.isChecked():
            type = 1
        else:
            type = 2

        self.Colonne_cortical_Thread = Colonne_cortical_Thread(self.Colonne, type=type)
        self.CC = self.Colonne_cortical_Thread.CC
        self.CC.updateTime.something_happened.connect(self.updateTime)
        self.createCells = True

        self.D_e.setText(str(self.CC.D))
        self.L_e.setText(str(self.CC.L))

        self.Layer_d1_l.setText(str(self.CC.Layer_d[0]))
        self.Layer_d23_l.setText(str(self.CC.Layer_d[1]))
        self.Layer_d4_l.setText(str(self.CC.Layer_d[2]))
        self.Layer_d5_l.setText(str(self.CC.Layer_d[3]))
        self.Layer_d6_l.setText(str(self.CC.Layer_d[4]))

        self.set_tissue_func()


        self.nbcellsnb1.setText(str(int(self.CC.Layer_nbCells[0])))
        self.nbcellsnb23.setText(str(int(self.CC.Layer_nbCells[1])))
        self.nbcellsnb4.setText(str(int(self.CC.Layer_nbCells[2])))
        self.nbcellsnb5.setText(str(int(self.CC.Layer_nbCells[3])))
        self.nbcellsnb6.setText(str(int(self.CC.Layer_nbCells[4])))

        self.update_cellNnumber()


    def LoadRes(self):
        pass
        # self.mascene_EEGViewer.update(Sigs_dict=self.Sigs_dict, Colors=self.Color)

    def setmarginandspacing(self, layout):
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)


class QuestionWhatToSave(QDialog):
    def __init__(self, parent=None, item=None, Graph_Items=None):
        super(QuestionWhatToSave, self).__init__(parent)
        self.parent = parent
        self.Param_box = QGroupBox("Select information you wan to save:")
        self.Param_box.setFixedWidth(300)
        self.layout_Param_box = QVBoxLayout()
        self.item = item

        self.CB_layout = QVBoxLayout()
        self.save_lfp_CB = QCheckBox('LFP')
        self.save_signal_CB = QCheckBox('Signals')
        self.save_coord_CB = QCheckBox('Coordinates')

        self.save_coord_l = QLabel('Coordinates can only be save in .mat and .data format.')
        self.CB_layout.addWidget(self.save_lfp_CB)
        self.CB_layout.addWidget(self.save_signal_CB)
        self.CB_layout.addWidget(self.save_coord_CB)
        self.CB_layout.addWidget(self.save_coord_l)
        [cb.setChecked(True) for cb in [self.save_lfp_CB, self.save_signal_CB, self.save_coord_CB]]

        self.horizontalGroupBox_Actions = QWidget()
        self.horizontalGroupBox_Actions.setFixedSize(285, 80)
        self.layout_Actions = QHBoxLayout()
        self.Button_Ok = QPushButton('Ok')
        self.Button_Cancel = QPushButton('Cancel')
        self.Button_Ok.setFixedSize(66, 30)
        self.Button_Cancel.setFixedSize(66, 30)
        self.layout_Actions.addWidget(self.Button_Ok)
        self.layout_Actions.addWidget(self.Button_Cancel)
        self.horizontalGroupBox_Actions.setLayout(self.layout_Actions)

        self.layout_Param_box.addLayout(self.CB_layout)
        self.layout_Param_box.addWidget(self.horizontalGroupBox_Actions)
        self.Param_box.setLayout(self.layout_Param_box)
        self.setLayout(self.layout_Param_box)

        self.Button_Ok.clicked.connect(self.myaccept)
        self.Button_Cancel.clicked.connect(self.reject)

    def myaccept(self):
        self.accept()


class CMViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(CMViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')  # Figure()
        self.figure.subplots_adjust(left=0.03, bottom=0.02, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self):
        ConnectivityMatrix = self.parent.CC.Conx['connectivitymatrix']
        ExternalPreSynaptic_Cell_AMPA_DPYR = self.parent.CC.ExternalPreSynaptic_Cell_AMPA_DPYR
        ExternalPreSynaptic_Cell_AMPA_Th = self.parent.CC.ExternalPreSynaptic_Cell_AMPA_Th
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        ax = self.figure.add_subplot(111)
        if scipy.sparse.issparse(ConnectivityMatrix):
            im = ax.spy(self.ConnectivityMatrix, markersize=1)
        elif type(ConnectivityMatrix) == type([]):

            # raw=[]
            # col=[]
            # dat=[]
            #
            # for l in range(len(ConnectivityMatrix)):
            #     for c in ConnectivityMatrix[l]:
            #         raw.append(l)
            #         col.append(c)
            #         dat.append(1)
            #         nb+=1
            # s = sparse.coo_matrix((dat, (raw, col)), shape=(len(ConnectivityMatrix), len(ConnectivityMatrix)))
            # im = ax.spy(s,markersize=1)
            raw = []
            col = []
            colors = []
            nb = 0
            flat_list = [item for sublist in self.parent.List_Colors for item in sublist]
            for l in range(len(ConnectivityMatrix)):
                raw.append(-self.parent.CC.NB_Th - self.parent.CC.NB_DPYR - 2)
                col.append(l)
                colors.append(flat_list[l])

                for c in ConnectivityMatrix[l]:
                    raw.append(c)
                    col.append(l)
                    colors.append(flat_list[c])
                    nb += 1
            for l in range(len(ExternalPreSynaptic_Cell_AMPA_DPYR)):
                for c in range(len(ExternalPreSynaptic_Cell_AMPA_DPYR[l])):
                    raw.append(-ExternalPreSynaptic_Cell_AMPA_DPYR[l][c] - 1)
                    col.append(l)
                    colors.append("#000000")
            for l in range(len(ExternalPreSynaptic_Cell_AMPA_Th)):
                for c in range(len(ExternalPreSynaptic_Cell_AMPA_Th[l])):
                    raw.append(-ExternalPreSynaptic_Cell_AMPA_Th[l][c] - 1 - self.parent.CC.NB_DPYR)
                    col.append(l)
                    colors.append("#999999")
            if platform.system() in ["Windows", "Linux"]:
                ax.scatter(raw, col, c=colors, s=10)
            elif platform.system() == "Darwin":
                ax.scatter(raw, col, c=colors, s=5)

        else:
            im = ax.imshow(ConnectivityMatrix)
        if platform.system() in ["Windows", "Linux"]:
            ax.set_title('ConnectivityMatrix' + str(nb), fontdict={'fontsize': 10})
        elif platform.system() == "Darwin":
            ax.set_title('ConnectivityMatrix' + str(nb), fontdict={'fontsize': 5})

        ax.set_xlabel('Sources')
        ax.set_ylabel('Targets')

        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1, linewidth=1, edgecolor='y', facecolor='none')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[2] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[2] - 1, linewidth=1, edgecolor='b', facecolor='none')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[1] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[1] - 1, linewidth=1, edgecolor='g', facecolor='none')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[0] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[0] - 1, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)

        self.canvas.draw_idle()
        self.canvas.show()


class StimViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(StimViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')  # Figure()
        self.figure.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self, stim, th):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        if not stim == []:
            t = np.arange(stim.shape[1]) * self.parent.dt
            ax = self.figure.add_subplot(211)
            for i, s in enumerate(stim):
                ax.plot(t, s + i * float(self.parent.TH_i_inj_e.text()), c='#000000')
        if not th == []:
            t = np.arange(th.shape[1]) * self.parent.dt
            ax = self.figure.add_subplot(212)
            for i, s in enumerate(th):
                ax.plot(t, s + i * float(self.parent.i_inj_e.text()), c='#999999')

        self.canvas.draw_idle()
        self.canvas.show()



class LFPViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(LFPViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')
        self.figure.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def addLFP(self, lfp, shiftT=0.):
        self.LFP = self.parent.LFP[-1]
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT

        ax = self.figure.add_subplot(111)
        # ax.plot(t, self.LFP)
        ax.plot(t, lfp)

        self.canvas.draw_idle()
        self.canvas.show()

    def updatePSD(self, shiftT=0.):
        self.LFP = self.parent.LFP[-1]
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        fs = int(1 / self.parent.dt)

        ax = self.figure.add_subplot(111)

        # f, t, Sxx = signal.spectrogram(self.LFP, fs* 1000, return_onesided=True,nperseg=fs*50, noverlap=fs*49, nfft=None)
        f, t, Sxx = signal.spectrogram(self.LFP, fs * 1000, return_onesided=True, nperseg=fs * 20, noverlap=fs * 19,
                                       nfft=fs * 100)
        fmax = 500
        indfmax = np.where(f > fmax)[0][0]
        # ax.pcolormesh(t, f[:indfmax], 10*np.log10(Sxx[:indfmax,:]) )
        ax.pcolormesh(t, f[:indfmax], Sxx[:indfmax, :])

        ax.set_ylabel('Frequency [Hz]')

        ax.set_xlabel('Time [sec]')

        self.canvas.draw_idle()
        self.canvas.show()
    def clearfig(self):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        self.axe = self.figure.add_subplot(111)
        self.canvas.draw_idle()

    def addline(self,LFP, shiftT=0,s=''):
        # self.LFP = self.parent.LFP
        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        self.axe.plot(t, LFP,label=s)
        self.axe.legend()
        self.canvas.draw_idle()

    def update(self, shiftT=0.):
        self.LFP = self.parent.LFP[-1]

        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT

        ax = self.figure.add_subplot(111)
        ax.plot(t, self.LFP)

        self.canvas.draw_idle()

    def update_AP_detect(self, APs, colors, titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        ax = self.figure.add_subplot(111)
        for i, ap in enumerate(APs):
            ax.scatter(ap, np.ones(len(ap))*i, c=colors[i])
        ax.set_ylabel(r'Neuron Number')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)
        self.canvas.draw_idle()
        self.canvas.show()

    def update_AP_polar(self, Phase_flat,Nbin=100, Lim=1 , titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        ax = self.figure.add_subplot(111, projection='polar')

        n, bins = np.histogram(np.array(Phase_flat), Nbin)
        n = n / np.sum(n)
        ax.hist((bins[0:-1] + bins[1:]) / 2, weights=n, bins=bins, rwidth=1.)
        binscenter = (bins[0:-1] + bins[1:]) / 2
        moyenne = (np.sum(n * binscenter) + np.sum(n[::-1] * -binscenter)) / np.sum(2 * n)
        moyenne = scipy.stats.circmean(np.array(Phase_flat))

        PLV = np.abs(np.mean(np.exp(1j * np.array(Phase_flat))))

        ax.plot([0, moyenne], [0, np.max(n) * PLV], c='r')

        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, Lim)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        self.canvas.draw_idle()
        self.canvas.show()


    def update_synchro_thresholding(self, sig, sigbin_perc, shiftT=0., titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        # plt.figure()
        ax = self.figure.add_subplot(111)
        ax.plot(t, sig)
        ax.plot(t, sigbin_perc)
        ax.set_ylabel(r'% pyramid firing')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)
        self.figure.set_tight_layout(True)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_AutoCorrelation(self, sig, autocorr, shiftT=0., titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        # plt.figure()
        ax = self.figure.add_subplot(211)
        ax.plot(t, sig)
        ax.set_ylabel(r'% pyramid firing ')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)
        ax2 = self.figure.add_subplot(212)
        ax2.plot(np.arange(len(autocorr)) * (t[1] - t[0]) - len(sig) * (t[1] - t[0]), autocorr)
        ax2.set_ylabel(r'autocorrelation ')
        ax2.set_xlabel(r'Time (s)')
        self.figure.set_tight_layout(True)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay_scatter(self, Aps_centers, shiftT=0., titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        for i in range(len(Aps_centers)):
            ax.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay_boxplot(self, Aps_delays, shiftT=0., titre=''):

        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax.boxplot(Aps_delays_flatten)
        ax.set_ylabel(r'delays (ms)')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay(self, Aps_centers, Aps_delays, shiftT=0., titre=''):

        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax1 = self.figure.add_subplot(121)
        for i in range(len(Aps_centers)):
            ax1.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax1.set_ylabel(r'Nb pyramid cell')
        ax1.set_xlabel(r'Time (s)')
        ax1.set_title(titre)

        ax2 = self.figure.add_subplot(122)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax2.boxplot(Aps_delays_flatten)
        ax2.set_ylabel(r'delays (ms)')
        ax2.set_xlabel(r'Nb pyramid cell')
        ax2.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay2(self, Aps_centers, Aps_delays, Aps_gauss_fit, shiftT=0., titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax1 = self.figure.add_subplot(131)
        for i in range(len(Aps_centers)):
            ax1.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax1.set_ylabel(r'Nb pyramid cell')
        ax1.set_xlabel(r'Time (s)')

        ax2 = self.figure.add_subplot(132)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax2.boxplot(Aps_delays_flatten)
        ax2.set_ylabel(r'delays (ms)')
        ax2.set_xlabel(r'Nb pyramid cell')
        ax2.set_title(titre)

        ax3 = self.figure.add_subplot(133)
        # ax3.boxplot(Aps_gauss_fit)
        ax3.scatter(range(len(Aps_gauss_fit)), Aps_gauss_fit, c='k')
        ax3.set_ylabel(r'gauss fit sigma')
        ax3.set_xlabel(r'Nb pyramid cell')
        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_ISI_distance(self, IntegralI, titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(IntegralI, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_vanRossum_distance(self, vr_dist, titre='vanRossum_distance'):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(vr_dist, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_VictorPurpura_distance(self, vp_dist, titre='VictorPurpura_distance'):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(vp_dist, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    # def closeEvent(self, event):
    #     plt.close(self.figure)


class Rescalesize(QDialog):
    def __init__(self, parent=None, x=None, y=None, z=None):
        super(Rescalesize, self).__init__(parent)
        self.parent = parent
        self.x = x
        self.y = y
        self.z = z
        self.Param_box = QWidget()
        self.Param_box.setFixedWidth(400)
        self.layout_Param_box = QVBoxLayout()

        self.Layout_param = QHBoxLayout()
        self.scale_GB = QGroupBox("rescale")

        # tissue size
        self.tissue_scale_GB = QGroupBox(r'scale value')
        labelX, self.xs_e = Layout_grid_Label_Edit(label=['x'], edit=['1'])
        labelY, self.ys_e = Layout_grid_Label_Edit(label=['y'], edit=['1'])
        labelZ, self.zs_e = Layout_grid_Label_Edit(label=['z'], edit=['1'])
        self.xs_e = self.xs_e[0]
        self.ys_e = self.ys_e[0]
        self.zs_e = self.zs_e[0]
        grid = QGridLayout()
        self.tissue_scale_GB.setLayout(grid)
        # grid.setContentsMargins(0,0,0,0)
        grid.setSpacing(0)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 1, 0)
        grid.addWidget(labelZ, 2, 0)
        self.xs_e.textChanged.connect(lambda state, s='x': self.rescale_axe(s))
        self.ys_e.textChanged.connect(lambda state, s='y': self.rescale_axe(s))
        self.zs_e.textChanged.connect(lambda state, s='z': self.rescale_axe(s))
        self.Layout_param.addWidget(self.tissue_scale_GB)

        self.tissue_size_GB = QGroupBox(r'tissue size')
        labelX, self.x_e = Layout_grid_Label_Edit(label=['x'], edit=[x])
        labelY, self.y_e = Layout_grid_Label_Edit(label=['y'], edit=[y])
        labelZ, self.z_e = Layout_grid_Label_Edit(label=['z'], edit=[z])
        self.x_e = self.x_e[0]
        self.y_e = self.y_e[0]
        self.z_e = self.z_e[0]
        grid = QGridLayout()
        self.tissue_size_GB.setLayout(grid)
        # grid.setContentsMargins(0,0,0,0)
        grid.setSpacing(0)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 1, 0)
        grid.addWidget(labelZ, 2, 0)
        self.x_e.textChanged.connect(lambda state, s='x': self.resize_axe(s))
        self.y_e.textChanged.connect(lambda state, s='y': self.resize_axe(s))
        self.z_e.textChanged.connect(lambda state, s='z': self.resize_axe(s))
        self.Layout_param.addWidget(self.tissue_size_GB)

        self.horizontalGroupBox_Actions = QGroupBox("Actions")
        self.horizontalGroupBox_Actions.setFixedSize(285, 80)
        self.layout_Actions = QHBoxLayout()
        self.Button_Ok = QPushButton('Ok')
        self.Button_Cancel = QPushButton('Cancel')
        self.Button_Ok.setFixedSize(66, 30)
        self.Button_Cancel.setFixedSize(66, 30)
        self.layout_Actions.addWidget(self.Button_Ok)
        self.layout_Actions.addWidget(self.Button_Cancel)
        self.horizontalGroupBox_Actions.setLayout(self.layout_Actions)

        self.layout_Param_box.addLayout(self.Layout_param)
        self.layout_Param_box.addWidget(self.horizontalGroupBox_Actions)
        self.Param_box.setLayout(self.layout_Param_box)
        self.setLayout(self.layout_Param_box)

        self.Button_Ok.clicked.connect(self.myaccept)
        self.Button_Cancel.clicked.connect(self.reject)

    def myaccept(self):
        self.accept()

    def resize_axe(self, s):
        try:
            if s == 'x':
                self.xs_e.blockSignals(True)
                self.xs_e.setText(str(float(self.x_e.text()) / float(self.x)))
                self.xs_e.blockSignals(False)
            elif s == 'y':
                self.ys_e.blockSignals(True)
                self.ys_e.setText(str(float(self.y_e.text()) / float(self.y)))
                self.ys_e.blockSignals(False)
            elif s == 'z':
                self.zs_e.blockSignals(True)
                self.zs_e.setText(str(float(self.z_e.text()) / float(self.z)))
                self.zs_e.blockSignals(False)
        except:
            pass

    def rescale_axe(self, s):
        try:
            if s == 'x':
                self.x_e.blockSignals(True)
                self.x_e.setText(str(float(self.xs_e.text()) * float(self.x)))
                self.x_e.blockSignals(False)
            elif s == 'y':
                self.y_e.blockSignals(True)
                self.y_e.setText(str(float(self.ys_e.text()) * float(self.y)))
                self.y_e.blockSignals(False)
            elif s == 'z':
                self.z_e.blockSignals(True)
                self.z_e.setText(str(float(self.zs_e.text()) * float(self.z)))
                self.z_e.blockSignals(False)
        except:
            pass


class Afferences_Managment(QDialog):
    def __init__(self, parent):
        super(Afferences_Managment, self).__init__()

        self.parent = parent

        self.layoutparam = QVBoxLayout()
        self.List_Var_GB = QWidget()
        label_source = QLabel('Target')
        label_Target = QLabel('S\no\nu\nr\nc\ne')
        label_PYR1 = QLabel('PYR')
        label_PYR2 = QLabel('PYR')
        label_BAS1 = QLabel('BAS')
        label_BAS2 = QLabel('BAS')
        label_OLM1 = QLabel('OLM')
        label_OLM2 = QLabel('OLM')
        label_BIS1 = QLabel('BIS')
        label_BIS2 = QLabel('BIS')
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        grid.setContentsMargins(5, 5, 5, 5, )
        grid.addWidget(label_source, 0, 1, 1, 5, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(label_Target, 1, 0, 5, 1, Qt.AlignmentFlag.AlignVCenter)

        # grid.addWidget(label_PYR1, 2, 1)
        # grid.addWidget(label_BAS1, 3, 1)
        # grid.addWidget(label_OLM1, 4, 1)
        # grid.addWidget(label_BIS1, 5, 1)
        # grid.addWidget(label_PYR2, 1, 2)
        # grid.addWidget(label_BAS2, 1, 3)
        # grid.addWidget(label_OLM2, 1, 4)
        # grid.addWidget(label_BIS2, 1, 5)

        # self.List_Var = self.List_Var / 15
        self.List_Var_e = []
        for l in range(self.parent.CC.Afferences.shape[0]):
            for c in range(self.parent.CC.Afferences.shape[1]):
                edit = LineEdit_Int(str(self.parent.CC.Afferences[l, c]))
                self.List_Var_e.append(edit)
                grid.addWidget(edit, l + 2, c + 2)
        self.List_Var_GB.setLayout(grid)

        self.buttons = QDialogButtonBox(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal, self)
        self.buttons.accepted.connect(self.myaccept)
        self.buttons.rejected.connect(self.reject)

        self.layoutparam.addWidget(self.List_Var_GB)
        self.layoutparam.addWidget(self.buttons)
        self.setLayout(self.layoutparam)

    def myaccept(self):
        matrice = np.zeros(self.parent.CC.Afferences.shape)
        ind = 0
        for l in range(self.parent.CC.Afferences.shape[0]):
            for c in range(self.parent.CC.Afferences.shape[1]):
                matrice[l, c] = float(self.List_Var_e[ind].text())
        self.parent.CC.Afferences = matrice * 0.5
        self.accept()


class Afferences_ManagmentTable(QDialog):
    def __init__(self, parent):
        super(Afferences_ManagmentTable, self).__init__()
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.parent = parent

        self.layoutparam = QVBoxLayout()
        self.tableWidget = QTableWidget()

        fnt = QFont()
        fnt.setPointSize(8)
        self.tableWidget.setFont(fnt)

        Afferences = self.parent.CC.Afferences
        line, column = Afferences.shape

        self.tableWidget.horizontalHeader().setDefaultSectionSize(42)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)

        self.tableWidget.setRowCount(line + 4)
        self.tableWidget.setColumnCount(column + 2)

        self.tableWidget.setItemDelegate(Delegate())

        item_Layer = ["I", "II/III", "IV", "V", "VI"]
        item_type = ["PC", "PV", "SST", "VIP", "RLN"]  # d0a9ce
        item_color = ["#d0a9ce", "#c5e0b4", "#dae0f3", "#f8cbad", "#ffe699"]

        for j in range(len(item_Layer)):
            item = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(2 + j * 5, 0, 5, 1)
            self.tableWidget.setItem(2 + j * 5, 0, item)
            item2 = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(0, 2 + j * 5, 1, 5)
            self.tableWidget.setItem(0, 2 + j * 5, item2)

        item = QTableWidgetItem("Thalamus")
        self.tableWidget.setSpan(2 + 24 + 1, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 1, 0, item)
        item = QTableWidgetItem("Distant Cortex")
        self.tableWidget.setSpan(2 + 24 + 2, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 2, 0, item)
        item = QTableWidgetItem("Sources")
        self.tableWidget.setItem(1, 0, item)
        item = QTableWidgetItem("Targets")
        self.tableWidget.setItem(0, 1, item)

        ind = 2
        self.List_possible_connexion = np.array(
            [[1, 1, 1, 1, 0],  # PC -> PC,PV,SST,VIP ,RLN  affinites de connexion entre cellules
             [1, 1, 0, 0, 0],  # PV -> PC,PV,SST,VIP ,RLN
             [1, 1, 0, 1, 1],  # SST -> PC,PV,SST,VIP ,RLN
             [0, 0, 1, 0, 0],  # VIP --> PC,PV,SST,VIP ,RLN
             [1, 1, 1, 1, 1]  # RLN --> PC,PV,SST,VIP ,RLN
             ])
        for i in range(5):
            for j in range(len(item_type)):
                item = QTableWidgetItem(item_type[j])
                item2 = QTableWidgetItem(item_type[j])
                item.setBackground(QColor(item_color[j]))
                item2.setBackground(QColor(item_color[j]))

                self.tableWidget.setItem(ind, 1, item)
                self.tableWidget.setItem(1, ind, item2)
                ind += 1

        for c in range(Afferences.shape[0]):
            type1 = int(np.mod(c, 5))
            for l in range(Afferences.shape[1]):
                type2 = int(np.mod(l, 5))
                item = QTableWidgetItem(str(Afferences[c, l]))
                # if self.List_possible_connexion[type2,type1]:
                #     item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(c + 2, l + 2, item)

        # label_source = QLabel('Target')
        # label_Target = QLabel('S\no\nu\nr\nc\ne')
        # label_PYR1 = QLabel('PYR')
        # label_PYR2 = QLabel('PYR')
        # label_BAS1 = QLabel('BAS')
        # label_BAS2 = QLabel('BAS')
        # label_OLM1 = QLabel('OLM')
        # label_OLM2 = QLabel('OLM')
        # label_BIS1 = QLabel('BIS')
        # label_BIS2 = QLabel('BIS')
        # grid = QGridLayout()
        # grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        # grid.setContentsMargins(5, 5, 5, 5, )
        # grid.addWidget(label_source, 0, 1, 1, 5, Qt.AlignmentFlag.AlignHCenter)
        # grid.addWidget(label_Target, 1, 0, 5, 1, Qt.AlignmentFlag.AlignVCenter)
        #
        # # grid.addWidget(label_PYR1, 2, 1)
        # # grid.addWidget(label_BAS1, 3, 1)
        # # grid.addWidget(label_OLM1, 4, 1)
        # # grid.addWidget(label_BIS1, 5, 1)
        # # grid.addWidget(label_PYR2, 1, 2)
        # # grid.addWidget(label_BAS2, 1, 3)
        # # grid.addWidget(label_OLM2, 1, 4)
        # # grid.addWidget(label_BIS2, 1, 5)
        #
        #
        # # self.List_Var = self.List_Var / 15
        # self.List_Var_e = []
        # for l in range(self.parent.CC.Afferences.shape[0]):
        #     for c in range(self.parent.CC.Afferences.shape[1]):
        #         edit = LineEdit_Int(str(self.parent.CC.Afferences[l, c]))
        #         self.List_Var_e.append(edit)
        #         grid.addWidget(edit, l + 2, c + 2)
        # self.List_Var_GB.setLayout(grid)

        self.math_param = QHBoxLayout()
        self.value_LE = LineEdit('0')
        self.add_PB = QPushButton('+')
        self.sub_PB = QPushButton('-')
        self.time_PB = QPushButton('x')
        self.divide_PB = QPushButton('/')
        self.roundup_PB = QPushButton(u'\u00CE')
        self.Save_PB = QPushButton('Save')
        self.Load_PB = QPushButton('Load')
        self.math_param.addWidget(QLabel('Value'))
        self.math_param.addWidget(self.value_LE)
        self.math_param.addWidget(self.add_PB)
        self.math_param.addWidget(self.sub_PB)
        self.math_param.addWidget(self.time_PB)
        self.math_param.addWidget(self.divide_PB)
        self.math_param.addWidget(self.roundup_PB)
        self.math_param.addWidget(self.Save_PB)
        self.math_param.addWidget(self.Load_PB)
        self.add_PB.clicked.connect(lambda state, x='+': self.math_fun(x))
        self.sub_PB.clicked.connect(lambda state, x='-': self.math_fun(x))
        self.time_PB.clicked.connect(lambda state, x='x': self.math_fun(x))
        self.divide_PB.clicked.connect(lambda state, x='/': self.math_fun(x))
        self.roundup_PB.clicked.connect(lambda state, x=u'\u00CE': self.math_fun(x))
        self.Save_PB.clicked.connect(self.Save_fun)
        self.Load_PB.clicked.connect(self.Load_fun)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal, self)
        self.buttons.accepted.connect(self.myaccept)
        self.buttons.rejected.connect(self.reject)

        self.layoutparam.addWidget(self.tableWidget)
        self.layoutparam.addLayout(self.math_param)
        self.layoutparam.addWidget(self.buttons)
        self.setLayout(self.layoutparam)

    def Save_fun(self):
        # try:
        matrice = np.zeros(self.parent.CC.Afferences.shape)
        for l in range(self.parent.CC.Afferences.shape[0]):
            for c in range(self.parent.CC.Afferences.shape[1]):
                item = self.tableWidget.item(l + 2, c + 2)
                matrice[l, c] = float(item.text())
        # except:
        #     msg_cri(s='The values in the table are not compatible.\nPlease check them.')

        extension = "csv"
        fileName = QFileDialog.getSaveFileName(caption='Save Matrix', filter=extension + " (*." + extension + ")")
        if (fileName[0] == ''):
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName = (fileName[0] + '.' + extension, fileName[1])
        # try:
        if fileName[1] == extension + " (*." + extension + ")":
            np.savetxt(fileName[0], matrice, delimiter=";", fmt='%0.4f')
        # except:
        #     msg_cri(s='Impossible to save the file.\n')

    def Load_fun(self):
        fileName = QFileDialog.getOpenFileName(self, "Load Matrix", "", "csv (*.csv)")
        if fileName[0] == '':
            return
        if fileName[1] == "csv (*.csv)":
            matrice = np.loadtxt(fileName[0], delimiter=";")
            for l in range(matrice.shape[0]):
                for c in range(matrice.shape[1]):
                    item = self.tableWidget.item(l + 2, c + 2)
                    item.setText(str(matrice[l, c]))

    def math_fun(self, s=''):
        Items = self.tableWidget.selectedItems()
        for item in Items:
            if item.column() > 1 and item.row() > 1:
                # type1 = int(np.mod(item.column(), 5))
                # type2 = int(np.mod(item.row(), 5))
                # if self.List_possible_connexion[type2, type1]:
                value = float(self.value_LE.text())
                cell_val = float(item.text())
                if s == '+':
                    item.setText(str(cell_val + value))
                elif s == '-':
                    item.setText(str(cell_val - value))
                elif s == 'x':
                    item.setText(str(cell_val * value))
                elif s == '/':
                    if not value == 0:
                        item.setText(str(cell_val / value))
                elif s == u'\u00CE':
                    item.setText(str(np.ceil(cell_val)))

    def myaccept(self):
        try:
            matrice = np.zeros(self.parent.CC.Afferences.shape)
            for l in range(self.parent.CC.Afferences.shape[0]):
                for c in range(self.parent.CC.Afferences.shape[1]):
                    item = self.tableWidget.item(l + 2, c + 2)
                    matrice[l, c] = float(item.text())
        except:
            msg_cri(s='The values in the table are not compatible.\nPlease check them.')
        self.parent.CC.Afferences = matrice
        self.accept()


class Connection_ManagmentTable(QDialog):
    def __init__(self, parent):
        super(Connection_ManagmentTable, self).__init__()
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.parent = parent

        self.layoutparam = QVBoxLayout()
        self.tableWidget = QTableWidget()

        fnt = QFont()
        fnt.setPointSize(8)
        self.tableWidget.setFont(fnt)

        Afferences = self.parent.CC.inputpercent
        line, column = Afferences.shape

        self.tableWidget.horizontalHeader().setDefaultSectionSize(42)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)

        self.tableWidget.setRowCount(line + 4)
        self.tableWidget.setColumnCount(column + 2)

        self.tableWidget.setItemDelegate(Delegate())

        item_Layer = ["I", "II/III", "IV", "V", "VI"]
        item_type = ["PC", "PV", "SST", "VIP", "RLN"]  # d0a9ce
        item_color = ["#d0a9ce", "#c5e0b4", "#dae0f3", "#f8cbad", "#ffe699"]

        for j in range(len(item_Layer)):
            item = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(2 + j * 5, 0, 5, 1)
            self.tableWidget.setItem(2 + j * 5, 0, item)
            item2 = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(0, 2 + j * 5, 1, 5)
            self.tableWidget.setItem(0, 2 + j * 5, item2)

        item = QTableWidgetItem("Thalamus")
        self.tableWidget.setSpan(2 + 24 + 1, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 1, 0, item)
        item = QTableWidgetItem("Distant Cortex")
        self.tableWidget.setSpan(2 + 24 + 2, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 2, 0, item)
        item = QTableWidgetItem("Sources")
        self.tableWidget.setItem(1, 0, item)
        item = QTableWidgetItem("Targets")
        self.tableWidget.setItem(0, 1, item)

        ind = 2
        for i in range(5):
            for j in range(len(item_type)):
                item = QTableWidgetItem(item_type[j])
                item2 = QTableWidgetItem(item_type[j])
                item.setBackground(QColor(item_color[j]))
                item2.setBackground(QColor(item_color[j]))

                self.tableWidget.setItem(ind, 1, item)
                self.tableWidget.setItem(1, ind, item2)
                ind += 1

        for c in range(Afferences.shape[0]):
            for l in range(Afferences.shape[1]):
                item = QTableWidgetItem(str(Afferences[c, l]))
                self.tableWidget.setItem(c + 2, l + 2, item)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.Save_PB = QPushButton('Save')
        self.Save_PB.clicked.connect(self.Save_fun)

        self.layoutparam.addWidget(self.tableWidget)
        self.layoutparam.addWidget(self.Save_PB)
        self.layoutparam.addWidget(self.buttons)
        self.setLayout(self.layoutparam)

    def Save_fun(self):
        # try:
        matrice = np.zeros(self.parent.CC.inputpercent.shape)
        for l in range(self.parent.CC.inputpercent.shape[0]):
            for c in range(self.parent.CC.inputpercent.shape[1]):
                item = self.tableWidget.item(l + 2, c + 2)
                matrice[l, c] = float(item.text())
        # except:
        #     msg_cri(s='The values in the table are not compatible.\nPlease check them.')

        extension = "csv"
        fileName = QFileDialog.getSaveFileName(caption='Save Matrix', filter=extension + " (*." + extension + ")")
        if (fileName[0] == ''):
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName = (fileName[0] + '.' + extension, fileName[1])
        # try:
        if fileName[1] == extension + " (*." + extension + ")":
            np.savetxt(fileName[0], matrice, delimiter=";", fmt='%0.4f')
        # except:
        #     msg_cri(s='Impossible to save the file.\n')

class Delegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(Delegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter
    # def sizeHint(self, option, index):
    #     s = QStyledItemDelegate.sizeHint(self, option, index)
    #     return max(s.width(), s.height()) * QSize(1, 1)
    # def createEditor(self, parent, option, index):
    #     editor = LineEdit(parent)
    #     return editor


def msg_wait(s):
    msg = QMessageBox()
    msg.setIconPixmap(QPixmap(os.path.join('icons', 'wait.gif')).scaledToWidth(100))
    icon_label = msg.findChild(QLabel, "qt_msgboxex_icon_label")
    movie = QMovie(os.path.join('icons', 'wait.gif'))
    setattr(msg, 'icon_label', movie)
    icon_label.setMovie(movie)
    movie.start()

    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setModal(False)
    # msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.show()
    return msg


def msg_cri(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()


def questionsure(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Warning)
    strname = s
    msg.setText(strname)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
    ok = msg.exec()
    if ok == QMessageBox.StandardButton.Ok:
        return True
    else:
        return False


def main():
    app = QApplication(sys.argv)

    # if platform.system() in ['Darwin', 'Linux']:
    #     app.setStyle('Windows')

    import logging
    logging.basicConfig(filename='../logdata.log', filemode='w', level=logging.ERROR)

    def my_excepthook(type, value, tback):
        # log the exception here
        co = tback.tb_frame.f_code
        func_name = co.co_name
        line_no = tback.tb_frame.f_lineno
        filename = co.co_filename
        logging.error("main crashed. Error: %s\n%s\n%s", type, value, tback)
        logging.error('Tracing exception: %s "%s" \non line %s of file %s function %s' % (
            type.__name__, value, line_no, filename, func_name))
        import traceback
        string = traceback.format_stack(tback.tb_frame)
        for s in string:
            logging.error('%s\n' % (s))
        # then call the default handler
        sys.__excepthook__(type, value, tback)
        sys.exit(app.exec())

    sys.excepthook = my_excepthook

    ex = ModelMicro_GUI(app)
    ex.setWindowTitle('Model micro GUI')
    # ex.showMaximized()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

