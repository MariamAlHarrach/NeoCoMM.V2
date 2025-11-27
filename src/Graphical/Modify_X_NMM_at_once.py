__author__ = 'Maxime'
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import numpy as np
import sys
import os
from scipy.spatial import distance

def Layout_grid_Label_Edit(label = ['None'],edit =['None']):
    widget = QWidget()
    layout_range = QVBoxLayout()
    # layout_range.setContentsMargins(5,5,5,5)

    grid = QGridLayout()
    grid.setContentsMargins(5,5,5,5)
    widget.setLayout(grid)
    layout_range.addLayout(grid)
    Edit_List =[]
    for idx in range(len(label)):
        Label = QLabel(label[idx])
        Edit = LineEdit(edit[idx])
        grid.addWidget(Label, idx, 0)
        grid.addWidget(Edit, idx, 1)
        Edit_List.append(Edit)
    return widget, Edit_List

class LineEdit(QLineEdit):
    KEY = Qt.Key.Key_Return
    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegularExpressionValidator(QRegularExpression("[+-]?\\d*[\\.]?\\d+"))
        QREV.setLocale(QLocale(QLocale.Language.English))
        self.setValidator(QREV)

def msg_cri(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()

def Layout_groupbox_Label_Edit(labelGroup ='None', label = ['None'],edit =['None'] ,popName=[],popColor=[], width = 150 , height_add =0, height_per_line=20 ):
    widgetglobal = QWidget()
    layoutglobal = QVBoxLayout()
    layoutnamecolor = QHBoxLayout()
    Label = QLabel('Name')
    Nameparam = QLineEdit(popName)
    colorbutton = QPushButton('')
    set_QPushButton_background_color(colorbutton, QColor(popColor))
    colorbutton.clicked.connect(lambda state, x=colorbutton: label_color_clicked(state, x))
    layoutnamecolor.addWidget(Label)
    layoutnamecolor.addWidget(Nameparam)
    layoutnamecolor.addWidget(colorbutton)

    layout = QGroupBox(labelGroup)
    if not width == None:
        layout.setFixedWidth(width)
    layout.setAlignment( Qt.AlignmentFlag.AlignTop)
    layout_range = QVBoxLayout()
    # layout.setFixedHeight( height_add + height_per_line* len(label))
    grid = QGridLayout()
    layout_range.addLayout(grid)
    Edit_List =[]
    for idx in range(len(label)):
        Label = QLabel(label[idx])
        Label.setFixedHeight(height_per_line)
        Edit = QLineEdit(edit[idx])
        Edit.setFixedHeight(height_per_line)
        grid.addWidget(Label, idx, 0)
        grid.addWidget(Edit, idx, 1)
        Edit_List.append(Edit)
    layout.setLayout(layout_range)

    scroll = QScrollArea()
    widget = QWidget()
    widget.setLayout(QHBoxLayout())
    widget.layout().addWidget(layout)
    scroll.setWidget(widget)
    # scroll.setWidgetResizable(True)
    scroll.setFixedWidth(width+75)
    scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
    layoutglobal.addLayout(layoutnamecolor)
    layoutglobal.addWidget(scroll)
    layoutglobal.setAlignment(Qt.AlignmentFlag.AlignTop)
    widgetglobal.setLayout(layoutglobal)
    widgetglobal.setFixedWidth(width + 100)
    return widgetglobal, Edit_List, Nameparam, colorbutton

def set_QPushButton_background_color(button=None, color=None):
    if color==None or button==None :
        return
    else :
        button.setAutoFillBackground(True)
        values = "{r}, {g}, {b} ".format(r = color.red(),
                                     g = color.green(),
                                     b = color.blue())
        button.setStyleSheet("QPushButton { background-color: rgb("+values+"); }")

def label_color_clicked(event,button):
    color = QColor(button.palette().button().color())
    colordial = QColorDialog(color)
    colordial.exec()
    selectedcolor = colordial.currentColor()
    colordial.close()
    set_QPushButton_background_color(button,selectedcolor)
    pass


class Modify_X_NMM(QMainWindow):
    Mod_OBJ = pyqtSignal(list,list,list)
    Close_OBJ = pyqtSignal()
    updateVTK_OBJ = pyqtSignal(list)
    def __init__(self, parent=None,List_Neurone_type = [],Dict_Param = [],List_Names=[],List_Color=[],initcell = None ,CellPosition = None):
        super(Modify_X_NMM, self).__init__()
        self.isclosed = False
        self.List_Neurone_type = List_Neurone_type
        self.Dict_Param = Dict_Param
        self.List_Color = List_Color
        self.List_Names = List_Names
        self.initcell = initcell
        self.CellPosition = CellPosition

        ############variable utiles###########
        self.height_per_line = 20
        self.height_add = 30
        self.width_per_col =155
        self.width_label =100
        self.Heigtheach =  600
        #######################################
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainHBOX_param_scene = QHBoxLayout()
        self.setMinimumHeight(600)
        # self.setMinimumWidth(800)

        # set Tabs
        self.set_param()
        self.mainHBOX_param_scene.addWidget(self.praram)
        self.centralWidget.setLayout(self.mainHBOX_param_scene)

    def set_param(self):

        self.praram = QGroupBox('Setup')
        # self.praram.setFixedHeight(self.Heigtheach)


        self.layout_setup = QHBoxLayout()

        #title


        self.layout_loadedpop = QVBoxLayout()

        # tissue size
        self.getClosestCell_GB = QGroupBox(r'select closest cell from')
        x = np.mean(self.CellPosition, axis=0)
        labelX, self.xs_e = Layout_grid_Label_Edit(label=['x'], edit=[str(x[0])])
        labelY, self.ys_e = Layout_grid_Label_Edit(label=['y'], edit=[str(x[1])])
        labelZ, self.zs_e = Layout_grid_Label_Edit(label=['z'], edit=[str(x[2])])
        self.xs_e = self.xs_e[0]
        self.ys_e = self.ys_e[0]
        self.zs_e = self.zs_e[0]
        self.getClosestCell_PB = QPushButton('Select')
        grid = QGridLayout()
        self.getClosestCell_GB.setLayout(grid)
        # grid.setContentsMargins(0,0,0,0)
        grid.setSpacing(0)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 0, 1)
        grid.addWidget(labelZ, 0, 2)
        grid.addWidget(self.xs_e, 0, 3)
        grid.addWidget(self.ys_e, 0, 4)
        grid.addWidget(self.zs_e, 0, 5)
        grid.addWidget(self.getClosestCell_PB, 1, 0,1,6)
        self.getClosestCell_PB.clicked.connect(self.getClosestCell_fun)


        Stimulation_title = QLabel('Extract from ')
        self.PopNumber = QComboBox()
        for txt in range(len(self.List_Names)):
            self.PopNumber.addItem(str(txt) + ' ' + self.List_Names[txt])
        self.PopNumber.currentIndexChanged.connect(self.update_combobox_parameter)
        #nvariable setting


        edit =[]
        list_variable = []
        for key, value in self.Dict_Param[self.initcell].items():
            edit.append(str(value))
            list_variable.append(key)
        self.layout_NMM_var, self.Edit_List_NMM_var,self.Nameparam, self.colorbutton  = Layout_groupbox_Label_Edit(labelGroup ='List of variables', label = list_variable,edit =edit,popName=self.List_Names[self.initcell],popColor=self.List_Color[self.initcell], width = 150)

        self.layout_loadedpop.addWidget(self.getClosestCell_GB)
        self.layout_loadedpop.addWidget(Stimulation_title)
        self.layout_loadedpop.addWidget(self.PopNumber)
        self.layout_loadedpop.addWidget(self.layout_NMM_var)

        #nvariable setting
        N = len(self.Dict_Param)
        sqrt_N = int(np.sqrt(N))
        nb_column = int(np.ceil(N / sqrt_N))
        if nb_column > 13:
            nb_column = 13

        nb_line = int(np.ceil(N / nb_column))


        layout_toApply = QGroupBox('Population to apply')
        grid = QGridLayout()
        layout_toApply.setLayout(grid)
        self.list_pop = []
        for l in np.arange(nb_line):
            for c in np.arange(nb_column):
                idx = (l)*nb_column + c +1
                if idx <= N:
                    CB  = QCheckBox(str(idx-1)+' '+self.List_Names[idx-1])
                    # CB.setFixedWidth(self.width_label/2)
                    if self.List_Neurone_type[idx-1] == self.List_Neurone_type[self.initcell]:
                        CB.setChecked(True)
                    else:
                        CB.setChecked(False)
                        CB.setEnabled(False)
                    grid.addWidget(CB, l, c)
                    self.list_pop.append(CB)
        scroll = QScrollArea()
        scroll.setWidget(layout_toApply)
        scroll.setWidgetResizable(True)

        #action
        widgetActions = QWidget()
        layout_Actions = QVBoxLayout()
        widgetActions.setLayout(layout_Actions)
        widgetActions.setFixedWidth(300)
        self.ClearAll = QPushButton('Clear All')
        self.ClearAll.setFixedWidth(self.width_label*1.5)
        self.ClearAll.clicked.connect(self.ClearAllclick)
        self.SelectAll = QPushButton('Select All')
        self.SelectAll.setFixedWidth(self.width_label*1.5)
        self.SelectAll.clicked.connect(self.SelectAllclick)

        self.grid_Selection_layoutFromTo = QHBoxLayout()
        grid_SelectionFromTo = QGridLayout()
        self.grid_Selection_layoutFromTo.addLayout(grid_SelectionFromTo)
        # from to
        self.FromTo_line_from_e = QLineEdit('')
        self.FromTo_line_from_e.setMinimumWidth(30)
        self.FromTo_line_from_e.setValidator(QIntValidator(0, 100000))
        self.FromTo_line_to_e = QLineEdit('')
        self.FromTo_line_to_e.setMinimumWidth(30)
        self.FromTo_line_to_e.setValidator(QIntValidator(0, 100000))
        self.FromTo_line_select = QPushButton('select')
        self.FromTo_line_unselect = QPushButton('unselect')
        self.FromTo_line_select.clicked.connect(lambda state, x='select_FromToline': self.select_FromTo(x))
        self.FromTo_line_unselect.clicked.connect(lambda state, x='unselect_FromToline': self.select_FromTo(x))
        line = 0
        col = 0
        grid_SelectionFromTo.addWidget(QLabel('from'), line, col)
        col += 1
        grid_SelectionFromTo.addWidget(self.FromTo_line_from_e, line, col)
        col += 1
        grid_SelectionFromTo.addWidget(QLabel('to'), line, col)
        col += 1
        grid_SelectionFromTo.addWidget(self.FromTo_line_to_e, line, col)
        col += 1
        grid_SelectionFromTo.addWidget(self.FromTo_line_select, line, col)
        col += 1
        grid_SelectionFromTo.addWidget(self.FromTo_line_unselect, line, col)

        self.grid_Selection_layoutifIn = QHBoxLayout()
        grid_SelectionifIn = QGridLayout()
        self.grid_Selection_layoutifIn.addLayout(grid_SelectionifIn)
        # from to
        self.ifIn_line_from_e = QLineEdit('')
        self.ifIn_line_from_e.setMinimumWidth(30)
        self.ifIn_line_select = QPushButton('select')
        self.ifIn_line_unselect = QPushButton('unselect')
        self.ifIn_line_select.clicked.connect(lambda state, x='select_ifInline': self.select_ifIn(x))
        self.ifIn_line_unselect.clicked.connect(lambda state, x='unselect_ifInline': self.select_ifIn(x))
        line = 0
        col = 0
        grid_SelectionifIn.addWidget(QLabel('If'), line, col)
        col += 1
        grid_SelectionifIn.addWidget(self.ifIn_line_from_e, line, col)
        col += 1
        grid_SelectionifIn.addWidget(QLabel('in name'), line, col)
        col += 1
        grid_SelectionifIn.addWidget(self.ifIn_line_select, line, col)
        col += 1
        grid_SelectionifIn.addWidget(self.ifIn_line_unselect, line, col)

        self.loadModelparam = QPushButton('Load model parameters')
        self.loadModelparam.setFixedWidth(self.width_label*1.5)
        self.loadModelparam.clicked.connect(self.loadModelparamclick)
        self.saveModelparam = QPushButton('Save model parameters')
        self.saveModelparam.setFixedWidth(self.width_label*1.5)
        self.saveModelparam.clicked.connect(self.saveModelparamclick)
        label = QLabel('')
        self.consider_nameandcolor = QRadioButton('Consider Name and color?')

        self.radiusselect = QGroupBox('select around selected cell')
        hlayout = QHBoxLayout()
        self.radiusselect.setLayout(hlayout)
        radius = QLabel(r'radius (µm)')
        self.radius_e = LineEdit('10')
        self.radiusApply_PB = QPushButton('select')
        self.radiusApply_PB.setFixedWidth(self.width_label*1.5)
        self.radiusApply_PB.clicked.connect(self.radiusApply_PBclick)
        hlayout.addWidget(radius)
        hlayout.addWidget(self.radius_e)
        hlayout.addWidget(self.radiusApply_PB)


        self.updateVTK = QPushButton('see selected cell onf graph')
        self.updateVTK.setFixedWidth(self.width_label * 3)
        self.updateVTK.clicked.connect(self.updateVTKclick)

        self.Apply = QPushButton('Apply')
        self.Apply.setFixedWidth(self.width_label*1.5)
        self.Apply.clicked.connect(self.Applyclick)
        self.updateVTK = QPushButton('see selected cell onf graph')
        self.updateVTK.setFixedWidth(self.width_label * 3)
        self.updateVTK.clicked.connect(self.updateVTKclick)
        layout_Actions.addWidget(self.ClearAll)
        layout_Actions.addWidget(self.SelectAll)
        layout_Actions.addLayout(self.grid_Selection_layoutFromTo)
        layout_Actions.addLayout(self.grid_Selection_layoutifIn)
        layout_Actions.addWidget(QLabel(''))
        layout_Actions.addWidget(self.loadModelparam)
        layout_Actions.addWidget(self.saveModelparam)
        layout_Actions.addWidget(label)
        layout_Actions.addWidget(self.consider_nameandcolor)
        layout_Actions.addWidget(QLabel(''))
        layout_Actions.addWidget(self.radiusselect)
        layout_Actions.addWidget(self.Apply)
        layout_Actions.addWidget(QLabel(''))
        layout_Actions.addWidget(self.updateVTK)
        layout_Actions.setAlignment(Qt.AlignmentFlag.AlignTop)


        self.layout_setup.addLayout(self.layout_loadedpop)
        self.layout_setup.addWidget(scroll)
        self.layout_setup.addWidget(widgetActions)
        self.praram.setLayout(self.layout_setup)

    def getClosestCell_fun(self):
        CellDistances = distance.cdist(self.CellPosition, [[float(self.xs_e.text()),float(self.ys_e.text()),float(self.zs_e.text())]], 'euclidean')
        self.PopNumber.setCurrentIndex(np.argmin(CellDistances))

    def update_combobox_parameter(self):
        idx = self.PopNumber.currentIndex()

        self.layout_loadedpop.removeWidget(self.layout_NMM_var)
        self.layout_NMM_var.close()
        edit = []
        list_variable = []
        for key, value in self.Dict_Param[idx].items():
            edit.append(str(value))
            list_variable.append(key)
        self.layout_NMM_var, self.Edit_List_NMM_var, self.Nameparam, self.colorbutton = Layout_groupbox_Label_Edit(labelGroup='List of variables', label=list_variable, edit=edit,
                                                                                                              popName=self.List_Names[self.initcell], popColor=self.List_Color[self.initcell],
                                                                                                              width=150)

        self.ui.gridLayout.addWidget(self.layout_NMM_var, 3, 0, 1, 1)
        self.layout_loadedpop.update()
        # self.layout_loadedpop.ad(self.layout_NMM_var)

        # i=0
        # for key, value in self.Dict_Param[idx].items():
        #     self.Edit_List_NMM_var[i].setText(str(value))
        #     i+=1
        # self.Nameparam.setText(self.List_Names[idx])
        # set_QPushButton_background_color(self.colorbutton, QColor(self.List_Color[idx]))

        # for id_cb, CB in enumerate(self.list_pop):
        #     if self.List_Neurone_type[idx] == self.List_Neurone_type[id_cb]:
        #         CB.setChecked(True)
        #         CB.setEnabled(True)
        #     else:
        #         CB.setChecked(False)
        #         CB.setEnabled(False)

    def select_FromTo(self,s):
            fr = int(self.FromTo_line_from_e.text())
            to = int(self.FromTo_line_to_e.text())
            if fr < to:
                if to >= len(self.list_pop):
                    to = len(self.list_pop)-1
                if s == 'select_FromToline' :
                    [cb.setChecked(True) for cb in self.list_pop[fr:to+1] if cb.isEnabled() ]
                elif s == 'unselect_FromToline':
                    [cb.setChecked(False) for cb in self.list_pop[fr:to+1] if cb.isEnabled() ]

    def select_ifIn(self,s):
        txt = self.ifIn_line_from_e.text()
        for idx, name in enumerate(self.List_Names):
            if txt in name and self.list_pop[idx].isEnabled():
                if s=='select_ifInline':
                    self.list_pop[idx].setChecked(True)
                elif s=='unselect_ifInline':
                    self.list_pop[idx].setChecked(False)

    def loadModelparamclick(self):
        model, modelname = self.Load_Model()
        if model == None:
            msg_cri("Unable to load model")
            return
        list_variable = list(self.Dict_Param[self.PopNumber.currentIndex()].keys())
        knownkey = []
        unknownkey = []
        for key in model:
            if key not in ['Name','Color']:
                if key not in list_variable:
                    unknownkey.append(key)
                else:
                    knownkey.append(key)
        if unknownkey:
            quit_msg = "The current NMM does not match the file\n" \
                       "unknown variables: " + ','.join([str(u) for u in unknownkey]) + "\n" \
                        "Do you want to load only the known parameters?" + "\n" \
                                                                                                                                           "known variables: " + ','.join(
                [str(u) for u in knownkey]) + "\n"
            reply = QMessageBox.question(self, 'Message',
                                         quit_msg, QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return


        for key, value in model.items():
            if key =='Name' :
                self.Nameparam.setText(value)
            if key =='Color':
                set_QPushButton_background_color(self.colorbutton, QColor(value))

            if key in list_variable:
                index = [i for i,x in enumerate(list_variable) if x==key]
                self.Edit_List_NMM_var[index[0]].setText(str(value))



    def Load_Model(self):
        extension = "txt"
        fileName = QFileDialog.getOpenFileName(caption='Load parameters', filter=extension + " (*." + extension + ")")
        if (fileName[0] == ''):
            return None, None
        if os.path.splitext(fileName[0])[1] == '':
            fileName = (fileName[0] + '.' + extension, fileName[1])
        if fileName[1] == extension + " (*." + extension + ")":
            f = open(fileName[0], 'r')
            line = f.readline()
            model = None
            modelname = None
            while not ("Model_info::" in line or line == ''):
                line = f.readline()
            if "Model_info" in line:
                model, modelname, line = self.read_model(f)
            f.close()
            return model, modelname

    def read_model(self, f):
        line = f.readline()
        if '=' in line:
            modelname = line.split('=')[-1]
            line = f.readline()
        else:
            modelname = ''
            line = f.readline()
        if '=' in line:
            nbmodel = int(line.split('=')[-1])
            line = f.readline()
        else:
            nbmodel = 1
            line = f.readline()

        numero = 0
        # if nbmodel >1 :
        #     numero = NMM_number(nbmodel)

        model = {}
        while not ("::" in line or line == ''):
            if not (line == '' or line == "\n"):
                lsplit = line.split("\t")
                name = lsplit[0]
                try:
                    val = float(lsplit[numero+1])
                except:
                    val = lsplit[numero+1]
                model[name] = val
            line = f.readline()
        return model, modelname, line




    def saveModelparamclick(self):
        extension = "txt"
        fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=extension + " (*." + extension + ")")
        if (fileName[0] == ''):
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName = (fileName[0] + '.' + extension, fileName[1])
        if fileName[1] == extension + " (*." + extension + ")":
            id_cell = self.PopNumber.currentIndex()
            val = []
            list_variable = []
            i=0
            for key, value in self.Dict_Param[id_cell].items():
                val.append(str(float(self.Edit_List_NMM_var[i].text())))
                list_variable.append(key)
                i+=1

            if self.consider_nameandcolor.isChecked() == True:
                val.append(self.Nameparam.text())
                val.append(self.colorbutton.palette().button().color().name())
                # val.append(self.colorbutton.palette().button().color().name().replace('#', ''))
                list_variable.append('Name')
                list_variable.append('Color')
            f = open(fileName[0], 'w')
            self.write_model(f, '', list_variable, val)
            f.close()

    def write_model(self, f, name, listVar, listVal):
        f.write("Model_info::\n")
        f.write("Model_Name = " + name + "\n")
        f.write("Nb_cell = " + str(1) + "\n")
        for idx_n, n in enumerate(listVar):
            f.write(n + "\t")
            f.write(str(listVal[idx_n]) + "\t")
            f.write("\n")







    def ClearAllclick(self):
        for cb in self.list_pop:
            cb.setChecked(False)

    def SelectAllclick(self):
        for cb in self.list_pop:
            cb.setChecked(True)

    def Applyclick(self):
        for idx, cb in enumerate(self.list_pop):
            if cb.isChecked():
                if self.consider_nameandcolor.isChecked():
                    self.List_Names[idx] = self.Nameparam.text()
                    self.List_Color[idx] = self.colorbutton.palette().button().color().name()

                for idx_v, key in enumerate(self.Dict_Param[idx].keys()):
                    self.Dict_Param[idx][key] = float(self.Edit_List_NMM_var[idx_v].text())

        self.Mod_OBJ.emit(self.Dict_Param,self.List_Names,self.List_Color)

    def updateVTKclick(self):
        selected_list = []
        for i, l in enumerate(self.list_pop):
            if l.isChecked():
                selected_list.append(i)
        self.updateVTK_OBJ.emit(selected_list)

    def radiusApply_PBclick(self):
        radius = float(self.radius_e.text())
        idx = self.PopNumber.currentIndex()
        coordinate = self.CellPosition[idx]
        distances = distance.cdist(self.CellPosition, [coordinate], 'euclidean')

        for i, d in enumerate(distances):
            if d<= radius:
                if self.list_pop[i].isEnabled():
                    self.list_pop[i].setChecked(True)





    def closeEvent(self, event):
        self.Close_OBJ.emit()
        self.isclosed = True
        self.close()




# def main():
#     app = QApplication(sys.argv)
#     pop = []
#     for idx in range(10):
#         pop.append(Model_Siouar_TargetGains.pop_Siouar())
#         pop[idx].random_seeded(10)
#     pop[0].A=10
#     pop[1].A=5
#     list_variable = Model_Siouar_TargetGains.get_Variable_Names()
#
#     ex = Modify_X_NMM(app,pop=pop,list_variable=list_variable)
#     ex.setWindowTitle('Create Connectivity Matrix')
#     # ex.showMaximized()
#     ex.show()
#     #ex.move(0, 0)
#     sys.exit(app.exec( ))
#
#
# if __name__ == '__main__':
#     main()