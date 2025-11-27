__author__ = 'Mariam'

import numpy as np
import vtk


#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class ElectrodeNN:

    def __init__(self, Cellpos, NBPYRL,NB_PV,NB_SST, NB_VIP,NB_RLN,re=50,Electrodepos=[-150,150, 400]):
        # self.PYRs = Pyrpos[0, :, :]
        # self.PYRHs = Pyrposh[0, :, :]
        # self.Bass = Bas[0, :, :]
        # self.Biss = Bis[0, :, :]
        # self.Som = SOM[0, :, :]
        # self.cm=cm
        self.cellpos = Cellpos
        self.NBPYR =NBPYRL
        self.PV=NB_PV
        self.SST= NB_SST
        self.VIP = NB_VIP
        self.RLN=NB_RLN
        self.r = re
        self.Electrodepos=Electrodepos
        self.List_of_lines = []
        self.List_of_lines_mappers = []
        self.List_of_lines_actors = []

    def cylinder_object(self, Epos=None, my_color="slategray"):
        if Epos is None:
            Epos = self.Electrodepos
        endPoint = np.add(Epos, np.array([0, 0, 400]))
        colors = vtk.vtkNamedColors()
        # Create a cylinder.
        # Cylinder height vector is (0,1,0).
        # Cylinder center is in the middle of the cylinder
        cylinderSource = vtk.vtkCylinderSource()
        cylinderSource.SetRadius(self.r)
        cylinderSource.SetResolution(50)

        # Generate a random start and end point
        # startPoint = [0] * 3
        # endPoint = [0] * 3

        rng = vtk.vtkMinimalStandardRandomSequence()
        rng.SetSeed(8775070)  # For testing.8775070

        # Compute a basis
        normalizedX = [0] * 3
        normalizedY = [0] * 3
        normalizedZ = [0] * 3

        # The X axis is a vector from start to end
        vtk.vtkMath.Subtract(endPoint, Epos, normalizedX)
        length = vtk.vtkMath.Norm(normalizedX)
        vtk.vtkMath.Normalize(normalizedX)

        # The Z axis is an arbitrary vector cross X
        arbitrary = [0] * 3
        for i in range(0, 3):
            rng.Next()
            arbitrary[i] = rng.GetRangeValue(-10, 10)
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        vtk.vtkMath.Normalize(normalizedZ)

        # The Y axis is Z cross X
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtk.vtkMatrix4x4()
        # Create the direction cosine matrix
        matrix.Identity()
        for i in range(0, 3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])
        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(Epos)  # translate to starting point
        transform.Concatenate(matrix)  # apply direction cosines
        transform.RotateWXYZ(90, 0, 0, 1)
        transform.Scale(1.0, length, 1.0)  # scale along the height vector
        transform.Translate(0, 0, 0)  # translate to start of cylinder

        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(cylinderSource.GetOutputPort())

        # Create a mapper and actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        mapper.SetInputConnection(cylinderSource.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d(my_color))
        return actor

    def PyrObject(self,pos,rp=7,hp=15, my_color="orchid_medium"):
        colors = vtk.vtkNamedColors()
        transform = vtk.vtkTransform()
        transform.RotateY(270)
        cone = vtk.vtkConeSource()
        cone.SetHeight(hp)
        cone.SetRadius(rp)
        cone.SetResolution(20)
        # Transform the polydata
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(transform)
        tf.SetInputConnection(cone.GetOutputPort())
        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        actor.SetPosition([pos[2], pos[1], pos[0]])
        mapper.SetInputConnection(cone.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d(my_color))
        return actor

    def BasObject(self, pos, r=7, my_color="DarkGreen"):
        colors = vtk.vtkNamedColors()
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(-pos[0], pos[1], pos[2])
        sphere.SetRadius(r)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d(my_color))
        return actor

    def render_scene(self, my_actor_list):
        renderer = vtk.vtkRenderer()
        for arg in my_actor_list:
            renderer.AddActor(arg)
        namedColors = vtk.vtkNamedColors()
        renderer.SetBackground(namedColors.GetColor3d("white"))

        window = vtk.vtkRenderWindow()
        window.SetWindowName("Oriented Cylinder")
        window.AddRenderer(renderer)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)

        # Visualize
        window.Render()
        interactor.Start()

    # Create the first line (between Origin and P0)
    def addconnections(self,x1,x2):
        colors = vtk.vtkNamedColors()
        my_color="white"
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(2)
        points.SetPoint(0, x1[0], x1[1], x1[2])
        points.SetPoint(1, x2[0], x2[1], x2[2])

        line0 = vtk.vtkLine()
        line0.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in the vtkPoints
        line0.GetPointIds().SetId(1, 1)  # the second 1 is the index of P0 in the vtkPoints

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line0)

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(linesPolyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d(my_color))
        return actor
    def addaxes(self):
        transform = vtk.vtkTransform()
        transform.Translate(-400, 0, 300)
        transform.Scale([100, 100, 100])
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleMode(vtk.vtkTextActor.TEXT_SCALE_MODE_NONE)
        axes.GetXAxisCaptionActor2D().GetTextActor().GetTextProperty().SetFontSize(25)
        axes.GetXAxisCaptionActor2D().GetTextActor().GetTextProperty().SetColor(0, 0, 0)
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleMode(vtk.vtkTextActor.TEXT_SCALE_MODE_NONE)
        axes.GetYAxisCaptionActor2D().GetTextActor().GetTextProperty().SetFontSize(25)
        axes.GetYAxisCaptionActor2D().GetTextActor().GetTextProperty().SetColor(0, 0, 0)
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleMode(vtk.vtkTextActor.TEXT_SCALE_MODE_NONE)
        axes.GetZAxisCaptionActor2D().GetTextActor().GetTextProperty().SetFontSize(25)
        axes.GetZAxisCaptionActor2D().GetTextActor().GetTextProperty().SetColor(0, 0, 0)
        return axes

    def plotCA1(self):
        rp=np.array([0,9,4,12,9])
        hp=np.array([0,6.32,7.57, 23.90,16.94])
        colors_PYR=['','mediumpurple','darkviolet','darkmagenta','indigo']
        my_list = []
        my_list.append(self.addaxes())
        # x = np.linspace(-150, 150, 4)
        # y = np.linspace(-150, 150, 4)
        # xx, yy = np.meshgrid(x, y)
        # c = np.zeros(shape=(3, 16))
        # c[0, :] = np.reshape(xx, (1, 16))
        # c[1, :] = np.reshape(yy, (1, 16))
        # c[2, :] = np.ones(shape=(1, 16))*150
        # for cpos in np.transpose(c):
        #     pos = cpos+np.array([-300,300,0])
        #     my_list.append(self.cylinder_object(pos,my_color="gold"))
        # my_list.append(self.cylinder_object())

        for i in range(len(self.cellpos)):
            layercell=self.cellpos[i]
            for j in range(0, self.NBPYR[i]):
                my_list.append(self.PyrObject(layercell[j], rp=rp[i],hp=hp[i],my_color=colors_PYR[i]))
        # for i in range(0, len(self.PYRHs)):
        #     my_list.append(self.PyrObject(self.PYRHs[i, :], my_color="Violet"))
            for j in range(self.NBPYR[i], self.NBPYR[i]+self.PV[i]):
                my_list.append(self.BasObject(layercell[j],my_color="ForestGreen"))
            for j in range(self.NBPYR[i]+self.PV[i],self.NBPYR[i]+self.PV[i]+ self.SST[i]):
                my_list.append(self.BasObject(layercell[j], r=6, my_color="Mediumblue"))
            for j in range(self.NBPYR[i]+self.PV[i]+self.SST[i],self.NBPYR[i]+self.PV[i]+self.SST[i]+ self.VIP[i]):
                my_list.append(self.BasObject(layercell[j], r=6, my_color="Indianred"))
            for j in range(self.NBPYR[i] + self.PV[i] + self.SST[i]+ self.VIP[i],self.NBPYR[i] + self.PV[i] + self.SST[i] + self.VIP[i]+self.RLN[i]):
                    my_list.append(self.BasObject(layercell[j], r=6, my_color="Orange"))
        #for i in range(len(self.cm[0][0][0])):
        sourcepos=self.cellpos[0][0]
        targetpos=self.cellpos[0][2]
 #       my_list.append(self.addconnections(sourcepos,targetpos))

        self.render_scene(my_list)

#
# CC=Column_morphology.Column(type=2)
# Pos=CreateColumn.PlaceCell_func(CC.L,CC.L_th,CC.D,CC.Layer_nbCells)
#
# drawNN = ElectrodeNN(Pos, CC.Layer_nbCells_pertype[0],  CC.Layer_nbCells_pertype[1], CC.Layer_nbCells_pertype[2],  CC.Layer_nbCells_pertype[3], CC.Layer_nbCells_pertype[4],Electrodepos=[-150,150,350])
# drawNN.plotCA1()