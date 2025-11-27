from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from scipy.spatial import distance
import platform


class Graph_EField_VTK(QMainWindow):
    def __init__(self, parent=None, fromfile = True ):
        super(Graph_EField_VTK, self).__init__(parent)

        if platform.system() == 'Darwin':
            self.facteurzoom = 1.05
        else:
            self.facteurzoom = 1.25

        self.parent = parent
        # self.scene = QGraphicsScene(self)
        # self.setScene(self.scene)

        self.frame = QFrame()

        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(.1, .1, .1)
        # light = self.ren.MakeLight()
        # light.SetAmbientColor(1, 1, 1)
        # light.SetDiffuseColor(1, 1, 1)
        # light.SetSpecularColor(1, 1, 1)
        # self.ren.AddLight(light)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)


        # interactor = vtk.vtkRenderWindowInteractor()
        # interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        # interactor.Start()


        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        # Create source
        self.scaling_x = 50
        self.scaling_y = 50
        self.scaling_z = 50

        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show()
        self.iren.Initialize()
        self.iren.Start()


        self.List_of_lines = []
        self.List_of_lines_mappers = []
        self.List_of_lines_actors = []
        self.List_of_forms = []
        self.List_of_forms_mappers = []
        self.List_of_forms_actors = []
        self.List_of_boundingbox_actors = []
        if fromfile:
            self.draw_Graph()
        else:
            self.draw_Graph_Const()

    def set_center(self):
        self.ren.ResetCamera()

    def draw_Shperes(self):
        if len(self.List_of_forms_actors) > 0:
            for actor in self.List_of_forms_actors:
                self.ren.RemoveActor(actor)

        self.List_of_forms = []
        self.List_of_forms_mappers = []
        self.List_of_forms_actors = []
        self.List_of_DiskPoints_actors=[]
        appendFilter = vtk.vtkAppendPolyData()

        for i in range(len(self.CellPosition)):
            layercell=self.CellPosition[i]
            for j in range(len(layercell)):
                color = QColor(self.List_Colors[i][j]).getRgb()
                # color = [c/255 for c in color[:3]]
                Colors = vtk.vtkUnsignedCharArray()
                Colors.SetNumberOfComponents(3)
                Colors.SetName("Colors")
                Colors.InsertNextTuple3(color[0], color[1], color[2])

                source = vtk.vtkSphereSource()
                source.SetCenter(layercell[j][0] * self.scaling_x, layercell[j][1] * self.scaling_y,
                                 layercell[j][2] * self.scaling_z)
                source.SetRadius(10.0 * 50)
                source.Update()
                Cellarray = source.GetOutput().GetPolys().GetNumberOfCells()
                for c in range(Cellarray):
                    Colors.InsertNextTuple3(color[0], color[1], color[2])

                source.GetOutput().GetCellData().SetScalars(Colors)
                source.Update()
                self.List_of_forms.append(source)
                appendFilter.AddInputData(source.GetOutput())

                # if self.List_Neurone_type[i][j] ==1:
                #     source = self.PyrObject(layercell[j], rp=rp[i],hp=hp[i],my_color=self.List_Colors[i][j])
                #     self.List_of_forms.append(source)
                #     appendFilter.AddInputData(source.GetOutput())
                # elif self.List_Neurone_type[i][j] ==2:
                #     source = self.BasObject(layercell[j],my_color=self.List_Colors[i][j])
                #     self.List_of_forms.append(source)
                #     appendFilter.AddInputData(source.GetOutput())
                # elif self.List_Neurone_type[i][j] ==3:
                #     source = self.BasObject(layercell[j], r=6, my_color=self.List_Colors[i][j])
                #     self.List_of_forms.append(source)
                #     appendFilter.AddInputData(source.GetOutput())
                # elif self.List_Neurone_type[i][j] ==4:
                #     source = self.BasObject(layercell[j], r=6, my_color=self.List_Colors[i][j])
                #     self.List_of_forms.append(source)
                #     appendFilter.AddInputData(source.GetOutput())
                # elif self.List_Neurone_type[i][j] ==5:
                #     source = self.BasObject(layercell[j], r=6, my_color=self.List_Colors[i][j])
                #     self.List_of_forms.append(source)
                #     appendFilter.AddInputData(source.GetOutput())


        appendFilter.Update()
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOn()
        mapper.SetInputConnection(cleanFilter.GetOutputPort())
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.ren.AddActor(actor)
        self.List_of_forms_actors.append(actor)


    def setScales(self,scalewidthfromGUI):
        self.scaling_x = scalewidthfromGUI
        self.scaling_y = scalewidthfromGUI
        self.scaling_z = scalewidthfromGUI

    def draw_axes(self):
        if len(self.List_of_axes_actors) > 0:
            for actor in self.List_of_axes_actors:
                self.ren.RemoveActor(actor)

        self.List_of_axes_actors = []
        transform = vtk.vtkTransform()
        transform.Translate(-self.scaling_x, -self.scaling_y, 0)
        axes = vtk.vtkAxesActor()
        # x = np.max(self.CellPosition,axis=0)
        x = (1,1,1)
        axes.SetTotalLength(x[0]* self.scaling_x, x[1]* self.scaling_y, x[2]* self.scaling_z)
        axes.SetUserTransform(transform)
        self.ren.AddActor(axes)

    def draw_BoundingBox(self):

        # sintheta = np.sin(self.EField_theta * np.pi * 2 / 360)
        # costheta = np.cos(self.EField_theta * np.pi * 2 / 360)
        #
        x = self.EField['x']
        y = self.EField['y']
        # z = self.EField['z']
        #
        # p = np.array([np.array([x[0],y[0],z[0]]),
        #      np.array([x[-1],y[0],z[0]]),
        #      np.array([x[0],y[-1],z[0]]),
        #      np.array([x[-1],y[-1],z[0]]),
        #      np.array([x[0],y[0],z[-1]]),
        #      np.array([x[-1],y[0],z[-1]]),
        #      np.array([x[0],y[-1],z[-1]]),
        #      np.array([x[-1],y[-1],z[-1]]),])
        # p[:,0],p[:,1] = p[:,0] * costheta - p[:,1] * sintheta , p[:,0] * sintheta + p[:,1] * costheta


        p0 = [np.min(x)* self.scaling_x, np.min(y)* self.scaling_y, self.EField['z'][0]* self.scaling_z]
        p1 = [np.max(x)* self.scaling_x, np.max(y)* self.scaling_y, self.EField['z'][-1]* self.scaling_z]



        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)

        # CubeSource = vtk.vtkCubeSource()
        # CubeSource.SetBounds(np.min(p[:,0])* self.scaling_x,
        #                      np.max(p[:,0])* self.scaling_x,
        #                      np.min(p[:,1])* self.scaling_y,
        #                      np.max(p[:,1])* self.scaling_y,
        #                      np.min(p[:,2])* self.scaling_z,
        #                      np.max(p[:,2])* self.scaling_z)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(lineSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0)
        actor.GetProperty().SetLineWidth(0)
        self.ren.AddActor(actor)


        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(lineSource.GetOutput())
        outline.Update()

        transform = vtk.vtkTransform()
        transform.RotateWXYZ(self.EField_theta , 0, 0, 1)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(outline.GetOutputPort())
        transformFilter.Update()


        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOn()
        mapper.SetInputConnection(transformFilter.GetOutputPort())
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetLineWidth(3)
        self.ren.AddActor(actor)

    def draw_BoundingBox_Const(self):
        xmin = 99999999
        xmax = -9999999
        ymin = 99999999
        ymax = -9999999
        zmin = 99999999
        zmax = -9999999
        for i in range(len(self.CellPosition)):
            mini = np.min(self.CellPosition[i],axis = 0)
            maxi = np.max(self.CellPosition[i],axis = 0)
            if xmin > mini[0]:
                xmin = mini[0]

            if ymin > mini[1]:
                ymin = mini[1]

            if zmin > mini[2]:
                zmin = mini[2]

            if xmax < maxi[0]:
                xmax = maxi[0]

            if ymax < maxi[1]:
                ymax = maxi[1]

            if zmax < maxi[2]:
                zmax = maxi[2]
        p0 = [xmin* self.scaling_x, ymin* self.scaling_y, zmin* self.scaling_z]
        p1 = [xmax* self.scaling_x, ymax* self.scaling_y, zmax* self.scaling_z]

        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(lineSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0)
        actor.GetProperty().SetLineWidth(0)
        self.ren.AddActor(actor)


        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(lineSource.GetOutput())
        outline.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOn()
        mapper.SetInputConnection(outline.GetOutputPort())
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetLineWidth(3)
        self.ren.AddActor(actor)

    def draw_Arrows(self):
        appendFilter = vtk.vtkAppendPolyData()
        sintheta = np.sin(self.EField_theta * np.pi * 2 / 360)
        costheta = np.cos(self.EField_theta * np.pi * 2 / 360)
        for i,x in enumerate(self.EField['x'][::20]):
            for j, y in enumerate(self.EField['y'][::20]):
                x2, y2 = x+0., y +0.
                xp = x2 * costheta - y2 * sintheta
                y2 = x2 * sintheta + y2 * costheta
                x2 = xp

                for k, z in enumerate(self.EField['z'][::20]):
                    E = self.EField['Er'][i,j,k,:]
                    E2 = E*1.
                    xp = E2[0] * costheta - E2[1] * sintheta
                    yp = E2[0] * sintheta + E2[1] * costheta
                    E2[0], E2[1]  = xp , yp




                    arrowSource = vtk.vtkArrowSource()
                    # arrowSource.SetShaftRadius(0.01)
                    # arrowSource.SetTipLength(.9)

                    startPoint = [x2* self.scaling_x,y2* self.scaling_y,z* self.scaling_z]
                    endPoint = [(x2+E2[0]*10)* self.scaling_x,(y2+E2[1]*10)* self.scaling_y,(z+E2[2]*10)* self.scaling_z]

                    normalizedX = [0 for i in range(3)]
                    normalizedY = [0 for i in range(3)]
                    normalizedZ = [0 for i in range(3)]

                    math = vtk.vtkMath()
                    math.Subtract(endPoint, startPoint, normalizedX)
                    length = math.Norm(normalizedX)
                    math.Normalize(normalizedX)
                    arbitrary = np.array([1, 1, 1])
                    math.Cross(normalizedX, arbitrary, normalizedZ)
                    math.Normalize(normalizedZ)
                    math.Cross(normalizedZ, normalizedX, normalizedY)

                    matrix = vtk.vtkMatrix4x4()
                    matrix.Identity()
                    for i in range(3):
                        matrix.SetElement(i, 0, normalizedX[i])
                        matrix.SetElement(i, 1, normalizedY[i])
                        matrix.SetElement(i, 2, normalizedZ[i])

                    transform = vtk.vtkTransform()
                    transform.Translate(startPoint)
                    transform.Concatenate(matrix)
                    transform.Scale(length, length, length)

                    transformPD = vtk.vtkTransformPolyDataFilter()
                    transformPD.SetTransform(transform)
                    transformPD.SetInputConnection(arrowSource.GetOutputPort())
                    transformPD.Update()
                    appendFilter.AddInputConnection(transformPD.GetOutputPort())

        appendFilter.Update()
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOn()
        mapper.SetInputConnection(cleanFilter.GetOutputPort())
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([1, 1, 1])
        self.ren.AddActor(actor)

    def draw_Arrows_Const(self):
        xmin = 99999999
        xmax = -9999999
        ymin = 99999999
        ymax = -9999999
        zmin = 99999999
        zmax = -9999999
        for i in range(len(self.CellPosition)):
            mini = np.min(self.CellPosition[i],axis = 0)
            maxi = np.max(self.CellPosition[i],axis = 0)
            if xmin > mini[0]:
                xmin = mini[0]

            if ymin > mini[1]:
                ymin = mini[1]

            if zmin > mini[2]:
                zmin = mini[2]

            if xmax < maxi[0]:
                xmax = maxi[0]

            if ymax < maxi[1]:
                ymax = maxi[1]

            if zmax < maxi[2]:
                zmax = maxi[2]

        appendFilter = vtk.vtkAppendPolyData()
        E = self.EField['Er']
        for x in np.linspace(xmin,xmax, 10):
            for y in np.linspace(ymin,ymax, 10):
                for z in np.linspace(zmin,zmax, 10):
                    arrowSource = vtk.vtkArrowSource()
                    # arrowSource.SetShaftRadius(0.01)
                    # arrowSource.SetTipLength(.9)

                    startPoint = [x* self.scaling_x,y* self.scaling_y,z* self.scaling_z]
                    endPoint = [(x+E[0]*10)* self.scaling_x,(y+E[1]*10)* self.scaling_y,(z+E[2]*10)* self.scaling_z]

                    normalizedX = [0 for i in range(3)]
                    normalizedY = [0 for i in range(3)]
                    normalizedZ = [0 for i in range(3)]

                    math = vtk.vtkMath()
                    math.Subtract(endPoint, startPoint, normalizedX)
                    length = math.Norm(normalizedX)
                    math.Normalize(normalizedX)
                    arbitrary = np.array([1, 1, 1])
                    math.Cross(normalizedX, arbitrary, normalizedZ)
                    math.Normalize(normalizedZ)
                    math.Cross(normalizedZ, normalizedX, normalizedY)

                    matrix = vtk.vtkMatrix4x4()
                    matrix.Identity()
                    for i in range(3):
                        matrix.SetElement(i, 0, normalizedX[i])
                        matrix.SetElement(i, 1, normalizedY[i])
                        matrix.SetElement(i, 2, normalizedZ[i])

                    transform = vtk.vtkTransform()
                    transform.Translate(startPoint)
                    transform.Concatenate(matrix)
                    transform.Scale(length, length, length)

                    transformPD = vtk.vtkTransformPolyDataFilter()
                    transformPD.SetTransform(transform)
                    transformPD.SetInputConnection(arrowSource.GetOutputPort())
                    transformPD.Update()
                    appendFilter.AddInputConnection(transformPD.GetOutputPort())

        appendFilter.Update()
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOn()
        mapper.SetInputConnection(cleanFilter.GetOutputPort())
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([1, 1, 1])
        self.ren.AddActor(actor)

    def draw_Graph(self,):
        self.List_Neurone_type = self.parent.List_Neurone_type
        self.List_Names = self.parent.List_Names
        self.List_Colors = self.parent.List_Colors
        self.CellPosition = self.parent.CellPosition

        self.NBPYR = self.parent.CC.NB_PYR
        self.PV = self.parent.CC.NB_PV
        self.SST = self.parent.CC.NB_SST
        self.VIP = self.parent.CC.NB_VIP
        self.RLN = self.parent.CC.NB_RLN

        self.EField = self.parent.EField
        self.EField_theta= self.parent.EField_theta

        self.ren.RemoveAllViewProps()

        # self.draw_Lines()
        self.draw_Shperes()
        self.draw_BoundingBox()
        self.draw_Arrows()
        # self.draw_axes()


        self.iren.GetRenderWindow().Render()

        self.set_center()

    def draw_Graph_Const(self):
        self.List_Neurone_type = self.parent.List_Neurone_type
        self.List_Names = self.parent.List_Names
        self.List_Colors = self.parent.List_Colors
        self.CellPosition = self.parent.CellPosition


        self.NBPYR = self.parent.CC.NB_PYR
        self.PV = self.parent.CC.C.NB_PV
        self.SST = self.parent.CC.C.NB_SST
        self.VIP = self.parent.CC.C.NB_VIP
        self.RLN = self.parent.CC.C.NB_RLN

        self.EField = self.parent.EField

        self.ren.RemoveAllViewProps()

        # self.draw_Lines()
        self.draw_Shperes()
        self.draw_BoundingBox_Const()
        self.draw_Arrows_Const()
        # self.draw_axes()


        self.iren.GetRenderWindow().Render()

        self.set_center()


    def Render(self, ):
        self.iren.GetRenderWindow().Render()

    def send_selected_cell(self):
        self.parent.update_ModXNMM_from_VTKgraph(self.selected_cells[0])


    def send_selected_cell_2(self):
        self.parent.NewModifyXNMM.ckick_from_VTK(self.selected_cells[0])




