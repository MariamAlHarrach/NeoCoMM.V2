


<img src="img/image1.png" alt="Figure" width="300"/>
  

**Manual**

**NeoCoMM:** Neocortical Computational Microscale Model

Table of Contents

[1\. Overview 2](#_Toc151722042)

[2\. How to install and open the software 2](#_Toc151722043)

[3\. The Graphical Interface 2](#_Toc151722044)

[**3.1.** The Simulation tuning 3](#_Toc151722045)

[1.1.1. Choose Tissue type 3](#_Toc151722046)

[1.1.2. Choose tissue geometry 3](#_Toc151722047)

[1.1.3. Choose number of neurons by layer and their types 3](#_Toc151722048)

[1.1.4. Create afference matrix 4](#_Toc151722049)

[1.1.5. Select the shape of the tissue and compute the connectivity 6](#_Toc151722050)

[1.1.6. Define Pyramidal cell subtype 8](#_Toc151722051)

[1.1.7. Define the stimulation of distant cortex 8](#_Toc151722052)

[1.1.8. Define the stimulation of Thalamus 9](#_Toc151722053)

[1.1.9. Define the parameter of the simulation 9](#_Toc151722054)

[4\. Modify neuron instances 11](#_Toc151722055)

[5\. Views 20](#_Toc151722056)

[**5.1.** LE transmembrane voltage view: 20](#_Toc151722057)

[**5.2.** The LFP view: 22](#_Toc151722058)

[**5.3.** Tissue 3D view: 23](#_Toc151722059)

[**5.4.** connectivity view: 24](#_Toc151722060)

[**5.5.** stimulation view: 24](#_Toc151722061)

[**5.6.** Load/Save simulation 25](#_Toc151722062)

[6\. Tutorial 25](#_Toc151722063)

[**6.1.** From sratch 25](#_Toc151722064)

[**6.2.** From a save file 28](#_Toc151722065)

# Overview

This manual is intended to help any user to perform computational simulations at microscale of epileptiform events using NeoCoMM. Although it mainly focuses on interictal epileptic pattern simulations, other type of neural activities can be simulated by adjusting the neural network parameters.

# How to install and open the software

Clone the repository from <https://gitlab.univ-rennes1.fr/myochum/neocomm>

Or from Pypi : Pip install NeoCOMM

From the Terminal, type “python NeoComm.py”
 
<img src="img/image2.png" alt="Figure" width="800"/>

# The Graphical User Interface
 <img src="img/image3.png" alt="Figure" width="600"/>

The first left third of the screen is dedicated to the tuning of the simulation, the middle of the screen is dedicated to view the signal respond of the simulation (membrane potentials and LFP signals). The left third of the screen is dedicated to view the tissue, the connectivity and the stimulation.

The first time the tissue model is created can be long due to some python just in time compilation of the model.

## The Simulation tuning
 
 <img src="img/image4.png" alt="Figure" width="400"/>

In this part, every subsection can be clicked on and will unfold the corresponding layout.

1.  

### Choose Tissue type
 
 <img src="img/image5.png" alt="Figure" width="400"/>

Choose if the tissue is Human, Rat or Mouse

### Choose tissue geometry

 <img src="img/image6.png" alt="Figure" width="400"/>


Define the geometry of the tissue, click on 
 <img src="img/image7.png" alt="Figure" width="200"/>
 to validate

### Choose number of neurons by layer and their types
 
 <img src="img/image8.png" alt="Figure" width="400"/>


Define the number of neurons in each layer. Click on Enter after entering a value to consider it. If you enter a total number of cell, this total number is split onto the number of cell in each layer by conserving the ratio between them. The repartition of neuron types can also be set here and you must click on 
 <img src="img/image9.png" alt="Figure" width="150"/>
 to apply them.

### Create afference matrix

This matrix is used to compute the synaptic connectivity among neurons. It defines how many connection there will be between one neurons type in a layer source toward one neurons type in a layer target (can be in the same layer).

 <img src="img/image10.png" alt="Figure" width="400"/>


Select <img src="img/image11.png" alt="Figure" width="200"/> if you want the matrix to be used as a percentage value of the total amount of neuron.

Select <img src="img/image12.png" alt="Figure" width="150"/> if you want to use the matrix as number of neuron directly

Click on <img src="img/image13.png" alt="Figure" width="100"/> allow you to modify the matrix within a new window

<img src="img/image14.png" alt="Figure" width="800"/>

The selected part can be modify by adding, subtract, multiply, divide by the number in the value field. The Î allow you to round up the selected matrix values. The matrix can be save and reload thank to the <img src="img/image15.png" alt="Figure" width="150"/> buttons.

Click on <img src="img/image16.png" alt="Figure" width="70"/> to apply the changes

Click on <img src="img/image17.png" alt="Figure" width="250"/> allow you to see exactly how many neurons are connected together (if the percentage of the number of cell is considered)

<img src="img/image18.png" alt="Figure" width="800"/>

### Select the shape of the tissue and compute the connectivity

<img src="img/image19.png" alt="Figure" width="400"/>

You can select different kind of geometry for the tissue : <img src="img/image20.png" alt="Figure" width="200"/>

The allow you to seed the result if the value is different of zero (the output of the placement and the connectivity matrix will always be the same)

<img src="img/image21.png" alt="Figure" width="100"/>will make the neurons placement in the tissue (it also automatically create the connection matrix), once done the 3D view to the right will display them:

<img src="img/image22.png" alt="Figure" width="300"/>

<img src="img/image23.png" alt="Figure" width="150"/>create the associated connectivity matrix and display it in the connectivity matrix view:

<img src="img/image24.png" alt="Figure" width="600"/>

The connectivity matrix can be computed without making a new cells placements. For instance if the afference matrix have changed without changing the cell number and type, then a connectivity matrix can be computed again.

### Define Pyramidal cell subtype

<img src="img/image25.png" alt="Figure" width="400"/>

Select the percentage of each PC subtype (each column must sum up to 1). Click on <img src="img/image26.png" alt="Figure" width="200"/>to consider the changes.


### E-Field parameter

<img src="img/image_Efield1.png" alt="Figure" width="400"/>

In this section, it is possible to set information about the E-Field that will be apply to the simulation.
 
<img src="img/image_Efield2.png" alt="Figure" width="150"/>allows to convert a txt file containing 3D matrix into the matlab file that the interface can use.
  
<img src="img/image_Efield3.png" alt="Figure" width="300"/>here, if the E-Field have be computed as a 3D matrix, it can be loaded as .mat file. 
 
<img src="img/image_Efield4.png" alt="Figure" width="250"/>select if the E-Field is coming from a file or generated in the GUI
 
<img src="img/image_Efield5.png" alt="Figure" width="400"/>for GUI generated software, we can defined the vector of the E-Field in the 3D space.
  
<img src="img/image_Efield6.png" alt="Figure" width="400"/>Time window where the E-Field will be applied in the simulation
 
<img src="img/image_Efield7.png" alt="Figure" width="250"/> selection of the type of signal the E-Field will take (4 possibility: Constant, Sinusoidal, rectangular or Triangular). The two other parameters are the magnitude (A) and the frequency (F)

<img src="img/image_Efield8.png" alt="Figure" width="80"/> Display the vector field in the 3D view of the cortical column 

<img src="img/image_Efield9.png" alt="Figure" width="400"/> 

### Define the stimulation of distant cortex

<img src="img/image27.png" alt="Figure" width="400"/>

Here the user can define the various parameters for the distant cortex stimulation. The seed here is apply also to the Thalamus stimulation in order to fix the simulation. If the seed is different of zero then the simulation will be the same.

### Define the stimulation of Thalamus

Configure the Thalamus stimulation

<img src="img/image28.png" alt="Figure" width="400"/>

### Define the parameter of the simulation

<img src="img/image29.png" alt="Figure" width="400"/>

Simulation duration is the time the simulation will last. The sampling frequency is given in kHz.

<img src="img/image30.png" alt="Figure" width="100"/>You may select a one shot stimulation (only one stimulation will be apply) or a periodic stimulation (the stimulation will be repeated every the given period (time in ms).

Stim Start allow to select the start of the stimulation (or the position of the stimulation is One shot is picked up). The stim stop is the end of the stimulation.

<img src="img/image31.png" alt="Figure" width="100"/>allows to display or not the 3D view of the neuron positions

<img src="img/image32.png" alt="Figure" width="100"/>allows you to display the transmembrane voltages or not. (to uncheck if the number of signals to plot is really huge)

<img src="img/image33.png" alt="Figure" width="200"/>select a certain percentage of the signal to plot. By default, 30% means that only 30% of each neuron types in each layer will be displayed.

<img src="img/image34.png" alt="Figure" width="100"/>this button should be used if the model itself or if the number of neuron or type have changed (from the tab “% of cell” for instance)

<img src="img/image35.png" alt="Figure" width="100"/>allows you to access to every instance of neurons in a single window (could be long to display if the number of neurons is hugee)

 <img src="img/image36.png" alt="Figure" width="600"/>

Every parameter could be tuned manually. Don’t forget to click on the apply button to confirm the changes you made.

<img src="img/image37.png" alt="Figure" width="100"/>allows you to modify several neuron instances at once. This is convenient when the number of neurons gets large.

<img src="img/image38.png" alt="Figure" width="100"/>reset all variable states of the model instances.

<img src="img/image39.png" alt="Figure" width="100"/>allows you to save the state of the last simulation (transmembrane voltages and PSPs)

<img src="img/image40.png" alt="Figure" width="400"/>

Position an electrode as a disk or a cylinder and apply the CSD computation by clicking on <img src="img/image41.png" alt="Figure" width="100"/>

<img src="img/image42.png" alt="Figure" width="600"/>

<img src="img/image43.png" alt="Figure" width="300"/> set the position of the center of the disk

<img src="img/image44.png" alt="Figure" width="300"/>set the radius and both inclination in degree

<img src="img/image45.png" alt="Figure" width="100"/>check to display the spectrogram instead of the temporal signal.

# Modify neuron instances

<img src="img/image46.png" alt="Figure" width="100"/>click on this button

To access to this window:

<img src="img/image47.png" alt="Figure" width="800"/>

If you click on a neuron in the 3D view it will appear with a box around it

<img src="img/image48.png" alt="Figure" width="300"/>

And it information will be automatically loaded in the left panel

<img src="img/image49.png" alt="Figure" width="300"/>

You can also from this panel chose a neuron by selected it on the combobox <img src="img/image50.png" alt="Figure" width="250"/>

From the right panel, you can select several cell with the same type of the selected cell (from the left panel)

<img src="img/image51.png" alt="Figure" width="300"/>

You can select on which layer the neurons will be selected:

Example by selecting every PC from layer 2 and 3

<img src="img/image52.png" alt="Figure" width="300"/>

<img src="img/image53.png" alt="Figure" width="600"/>
You can select neurons by their number

<img src="img/image54.png" alt="Figure" width="300"/>
<img src="img/image55.png" alt="Figure" width="600"/>

You can select the neurons by with a sting caracters in their names

Ex: 
<img src="img/image56.png" alt="Figure" width="400"/>

<img src="img/image57.png" alt="Figure" width="600"/>

<img src="img/image58.png" alt="Figure" width="400"/>

Ex <img src="img/image59.png" alt="Figure" width="400"/>

<img src="img/image60.png" alt="Figure" width="600"/>

<img src="img/image61.png" alt="Figure" width="400"/>

<img src="img/image62.png" alt="Figure" width="600"/>

<img src="img/image63.png" alt="Figure" width="400"/>
 
select a cube around the selected cell

<img src="img/image64.png" alt="Figure" width="600"/>

Once you selected the cell that you want to modify and set the new parameter of the model you want to apply, just click on the apply button. <img src="img/image65.png" alt="Figure" width="250"/>

If you don’t want to modify the names or the colors of the selected neurons, uncheck those checkbox <img src="img/image66.png" alt="Figure" width="100"/>

Every selected cell will have now the new parameter.

This particular button

<img src="img/image67.png" alt="Figure" width="300"/> allows you to modify one parameter for every selected cell without modified the others.

The parameter of the current neuron can be save and load in/from a file thanks to those two buttons <img src="img/image68.png" alt="Figure" width="300"/>


### Action potential detection

<img src="img/image_AP1.png" alt="Figure" width="400"/>

Possibility to detect action potentials on the Vm potentials of each neuron. Two methods are implemented, a threshold and a peakdet. 

Results are display in the LFP view 

<img src="img/image_AP1.png" alt="Figure" width="600"/>


### Display polar plot of the position of APs with respect to the phase of the E-Field stimulation

Is it possible to display the polar plot that make correspondence between the position of Aps in the simulation and the phase of the stimulation E-Field

<img src="img/image_Polar1.png" alt="Figure" width="400"/>
<img src="img/image_Polar2.png" alt="Figure" width="400"/>
 
# Views

## The transmembrane voltage view

<img src="img/image69.png" alt="Figure" width="600"/>

<img src="img/image70.png" alt="Figure" width="200"/> multiply the amplitude by the factor. Press Enter to validate (it is only a representation factor, the transmembrane voltage values remains the same)

<img src="img/image71.png" alt="Figure" width="200"/> time that is display on the screen. If the time is reduced, a slider bar appears below the view to let you navigate between time windows

<img src="img/image72.png" alt="Figure" width="800"/>

<img src="img/image73.png" alt="Figure" width="200"/> allow you to add space between the signal

For instance with 5 <img src="img/image74.png" alt="Figure" width="800"/>

A slider bar appears to the right of the plot to navigate vertically

<img src="img/image75.png" alt="Figure" width="200"/> change the thickness of the lines

<img src="img/image76.png" alt="Figure" width="100"/>allows you to select the signal you want to see

For instance<img src="img/image77.png" alt="Figure" width="800"/>

Will display only three signal

<img src="img/image78.png" alt="Figure" width="800"/>

<img src="img/image79.png" alt="Figure" width="100"/> will display the matplotlib toolbox
<img src="img/image80.png" alt="Figure" width="200"/> on the top of the view.

<img src="img/image81.png" alt="Figure" width="200"/> modifies the vertical gray line that is plot every some ms.

<img src="img/image82.png" alt="Figure" width="200"/> allow to filter the signal with various kind of filters:

<img src="img/image83.png" alt="Figure" width="700"/>
You have to select the kind of filter

<img src="img/image84.png" alt="Figure" width="150"/>

Then enter the corresponding parameters for that selected filter

<img src="img/image85.png" alt="Figure" width="800"/>

Then click <img src="img/image86.png" alt="Figure" width="100"/> to add you filter. Note that several filter can be set consecutively.

Once all filter have been set up, click on <img src="img/image87.png" alt="Figure" width="100"/> to validate

<img src="img/image88.png" alt="Figure" width="100"/> select to apply the filters

<img src="img/image89.png" alt="Figure" width="100"/> allows you to save the transmembrane voltage in different formats: 

<img src="img/image90.png" alt="Figure" width="150"/>

## The LFP view

Once the LFP is computed, it is display in the LFP view

<img src="img/image91.png" alt="Figure" width="600"/>

It is possible to also display the spectrogram

## Tissue 3D view

<img src="img/image92.png" alt="Figure" width="600"/>

The VTK view allows to see the 3D representation of the positions and types of neurons. It also represent the electrode position and shape.

<img src="img/image93.png" alt="Figure" width="250"/> set the radius for the neuron (only for the view)


<img src="img/image94.png" alt="Figure" width="150"/> apply a scaling on the 3D view

<img src="img/image95.png" alt="Figure" width="100"/> redraw the 3Dview

## connectivity view

<img src="img/image96.png" alt="Figure" width="600"/>

This view represent the synaptic connections that exist in the tissue. Each line represent a target (the neuron that receive the synaptic input) and each column represent the source (the neurons that send an synaptic output). The last vertical line to the left represent the color (the type) of neurons that receive the synaptic connections for a line. The gray dot represent the thalamic input connection, and the black dot represent the Distant cortex connections. Each other dots represents a connection inside the tissue and their color represent the type of the neuron source for that connection.

## stimulation view

After a simulation, the user can click on <img src="img/image97.png" alt="Figure" width="100"/> to display the stimulation signals that have been apply onto the Thalamus and the distant cortex.

<img src="img/image98.png" alt="Figure" width="600"/>

## Load/Save simulation

<img src="img/image99.png" alt="Figure" width="150"/> the simulation can be save in a text file and load to retrieve the simulation. If seed have been enter properly, the simulation will be exactly the same.

# Tutorial

## From sratch

Open the software

<img src="img/image100.png" alt="Figure" width="150"/>

Select a tissue type

<img src="img/image101.png" alt="Figure" width="400"/>

Enter a total number of 2002 cells en click Enter

<img src="img/image102.png" alt="Figure" width="400"/>

Choose a cylinder shape for the cortical column and a seed value of 10. Then press “Place cells”

<img src="img/image103.png" alt="Figure" width="400"/>

Wait for the placement to be done

<img src="img/image104.png" alt="Figure" width="200"/>

The first time the model is used, the compilation of the model occur, it can take a minute, but will no longer be compiled after (just in time compilation in python numba module)

<img src="img/image105.png" alt="Figure" width="150"/>

The view of the tissue and the connectivity are automatically updated

<img src="img/image106.png" alt="Figure" width="400"/>

Click on run <img src="img/image107.png" alt="Figure" width="100"/> to start the simulation

At the end the transmembran voltage view Is updated

<img src="img/image108.png" alt="Figure" width="600"/>

Click on <img src="img/image109.png" alt="Figure" width="100"/> to compute the LFP

It appear on the LFP view

<img src="img/image110.png" alt="Figure" width="600"/>

## From a save file

To load a file go in Load Simulation

<img src="img/image111.png" alt="Figure" width="150"/>

Select a file

<img src="img/image112.png" alt="Figure" width="600"/>

Then the placement, the connectivity matrix and the model creation is done automatically

The views are updated

<img src="img/image113.png" alt="Figure" width="400"/>

You may adapt the simulation from here

Click on <img src="img/image114.png" alt="Figure" width="100"/>to launch the simulation

Click on <img src="img/image115.png" alt="Figure" width="100"/>to display the LFP

<img src="img/image116.png" alt="Figure" width="600"/>