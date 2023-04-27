#!/usr/bin/env python
# coding: utf-8

import underworld as uw
from underworld import function as fn

import math
import time as timekeeper
import numpy
import numpy as np

import underworld.visualisation as vis
import matplotlib.pyplot as plt
from IPython import display


rank = uw.mpi.rank


# Set up parameters of model space
# ------
# 

# Set simulation box size.
boxHeight = 1.0
boxLength = 3.0
# Set the resolution.
res = 256   # make sure this resolution matches what you eventually will use for the other model.  
            # Otherwise you'll have to play some extrapolation tricks 
n=res
    
# Set min/max temperatures.
tempMin = 0.0
tempMax = 1.0


# Set up mesh
# -----------

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                 elementRes  = (int(boxLength*res), res), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (boxLength, boxHeight),
                                 periodic     = [True, False])

velocityField       = mesh.add_variable(         nodeDofCount=2 )
pressureField       = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField    = mesh.add_variable(         nodeDofCount=1 )
temperatureDotField = mesh.add_variable(         nodeDofCount=1 )

# Initialise values, Added temperatureField and temperatureDotField values to test temp tracer
velocityField.data[:]       = [0.,0.]
pressureField.data[:]       = 0.
temperatureField.data[:]    = 1.0
temperatureDotField.data[:] = 0.


# Let's Load Data from other model
# -------
# 

# Read temperature data
readTemperature = True
# Read swarm data
loadData = True

#determining the last step ran
#MAKE SURE THAT YOU UPLOADED the mesh & temperature files as well as the FrequentOutput.dat file AND RENAMED THE FOLDER TO Ra1e7isoOutput

input_dir = 'Ra1e5isoOutput/' #DO CHANGE THIS TO MATCH WHERE THE INPUT FILES ARE SAVED
if not loadData:
    step = 0
    time = 0.0
    rStep = -1.0
else:
    dataload = numpy.loadtxt(input_dir + 'FrequentOutput.dat', skiprows=4)
    nL = dataload[-1,0]
    nL = int(-1-(nL % 1000))
    step = int(dataload[nL,0])
    time = dataload[nL,1] 
    rStep = step

if readTemperature:
    temperatureField.load(input_dir + 'temperature_%i.h5' %step, interpolate=True)

else:  #this will set up a sinusoidal temp field
    pertStrength = 0.2
    deltaTemp = tempMax - tempMin
    for index, coord in enumerate(mesh.data):
        pertCoeff = math.cos( math.pi * coord[0] ) * math.sin( math.pi * coord[1] )
        temperatureField.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
        temperatureField.data[index] = max(tempMin, min(tempMax, temperatureField.data[index]))


# Bookkeeping
# -----
# Where should we keep our results?

outputPath = '1e5RectN1e4092TempTracerOutput/'  #you may want to change this
# Make output directory if necessary
if rank==0:
   import os
   if not os.path.exists(outputPath):
      os.makedirs(outputPath)

writefigures = True  #toggle to set whether to write figures to output directory
        
        
# Boundary & Initial Conditions
# ------

# Set top and bottom wall temperature boundary values.


for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = tempMax
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = tempMin


iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

BottomWall = mesh.specialSets["MinJ_VertexSet"] 
TopWall = mesh.specialSets["MaxJ_VertexSet"] 
LeftWall = mesh.specialSets["MinI_VertexSet"] 
RightWall = mesh.specialSets["MaxI_VertexSet"] 


# Construct sets for ``I`` (vertical) and ``J`` (horizontal) walls.

# Create Direchlet, or fixed value, boundary conditions. More information on setting boundary conditions can be found in the **Systems** section of the user guide.

# 2D velocity vector can have two Dirichlet conditions on each vertex, 
# using periodic in x BC for this set up

# make sure these match the boundary conditions you'll eventually use for the full model

velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                           indexSetsPerDof = (iWalls, jWalls) )

# Temperature is held constant on the jWalls
tempBC = uw.conditions.DirichletCondition( variable        = temperatureField, 
                                           indexSetsPerDof = (jWalls,) )


# Let's see if we got it correct...

figtemp = vis.Figure( figsize=(800,400) )
figtemp.append( vis.objects.Surface(mesh, temperatureField, colours="gray") )
#figtemp.append( vis.objects.Mesh(mesh) )
#toggle the above line on if you want to make sure your mesh is refined enough - you should see that there are three elements across the boundary layer
#figtemp.show()

temperatureFieldIntegral = uw.utils.Integral(fn = temperatureField,mesh= mesh,integrationType="volume")
volume_integral = uw.utils.Integral( mesh=mesh, fn=1., integrationType="volume" )
volume = volume_integral.evaluate()
avTemperature = temperatureFieldIntegral.evaluate()[0]/volume[0]
#print (avTemperature)

#note, the average temperature should be around 0.5 - or half of the maximum temperature difference in the model


# Set up material parameters and functions
# ----------
# 
# Set functions for viscosity, density and buoyancy force. These functions and variables only need to be defined at the beginning of the simulation, not each timestep.

# Starting with Materials and Swarm
# ------

swarmMaterials         = uw.swarm.Swarm( mesh=mesh )
#swarmLayout      = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
swarmLayout    =uw.swarm.layouts.PerCellSpaceFillerLayout (swarm=swarmMaterials, particlesPerCell=20)
swarmMaterials.populate_using_layout( layout=swarmLayout )

nParticles = 20

# particle population control object (has to be called)
population_control = uw.swarm.PopulationControl(swarmMaterials,
                                                aggressive=False,splitThreshold=0.15, maxDeletions=2,maxSplits=5,
                                                particlesPerCell=nParticles)

# Set up material swarms
materialIndex  = swarmMaterials.add_variable( dataType="int",    count=1 )

# all the potential materials.  for the reference case, let's just have average homogeneous lithospheric structure
continent1Material = 0
refmantleMaterial = 1

continent1Depth = 0.7
cont1X1 =  1.00       #use these when we want to embed continents of a certain size 
cont1X2 =  2.00

#give shapes a material

materialIndex.data[:] = refmantleMaterial

#straight margins

for index,coord in enumerate(swarmMaterials.particleCoordinates.data):  
     if coord[1] > continent1Depth and coord[0] > cont1X1 and coord[0] < cont1X2:
         materialIndex.data[index] = continent1Material

#diagonal shape
#for index,coord in enumerate(swarmMaterials.particleCoordinates.data):  
#    if  coord[1] >= -2.8*coord[0]+3.8 and coord[1] >= 2.8* coord[0]-4.6 and coord[1]>=continent1Depth: 
#        materialIndex.data[index] = continent1Material
#    if coord[0] > 1.5  and coord[1] > 2.8*coord[0]-5.6:
#        materialIndex.data[index] = continent1Material
    
    
materialPoints = vis.objects.Points(swarmMaterials, materialIndex, pointSize=3.,  colours='purple red green blue gray')
materialPoints.colourBar.properties = {"ticks" : 2, "margin" : 40,  "align" : "center"}


figMaterialMesh = vis.Figure(title="Materials and Mesh", quality=3)
#figMaterialMesh.append( glucifer.objects.Mesh(mesh) )
#again, toggle this back on to make sure your resolution is sufficient
figMaterialMesh.append( materialPoints )
#figMaterialMesh.show() 
        
        
# we'll assign values to the materials like rheology, density, etc, in a bit


# Setting Values to Materials
# -----

# There are two ways to implement a temperature dependent viscosity -  
# 
# using the Frank-Kamenetskii linearized viscosity (F-K):
# 
# $$
# \eta = \eta_0 \exp(-CT)
# $$
# 
# or the temperature / pressure dependent Arrhenius form:
# 
# $$
# \eta = \eta_0 \exp \left(  \frac{ E^* + p V^* }{ T+ T_0 } \right)
# $$
# 
# Right now the F-K is being used, but you could always comment it out and use the Arrhenius form if you wanted.
# 
# We can use an isoviscous approach where we turn off the temperature dependence and just use the viscosity contrast, in other words, the relative viscosity normalized to an average mantle viscosity.  


#Arrhenius viscosity
#eta0 = 1.0e-6
#activationEnergy = 27.63102112
#fn_viscosity = eta0 * fn.math.exp( activationEnergy / (temperatureField+1.) )


tempdepend = False #toggle to use temp depend viscosity

if tempdepend :
#F-K approximation
    surfEtaCont1 = 1.0e3    #highest viscosity for continents
    surfEtaMantle = 1.0e1  #highest viscosity for mantle
    cEtaCont1 = numpy.log(surfEtaCont1) / tempMax
    cEtaMantle = numpy.log(surfEtaMantle) / tempMax

else :
#isoviscous viscosity contrast approach    
    cEtaCont1 = 1e4
    cEtaMantle = 1.0


refcEtaMap  = {      continent1Material : cEtaCont1, 
                     refmantleMaterial : cEtaMantle }

refcEtaFn    = fn.branching.map( fn_key = materialIndex, mapping = refcEtaMap )

if tempdepend :
    fn_viscosity = uw.function.math.exp(refcEtaFn *(tempMax-temperatureField))

else :   
    fn_viscosity = refcEtaFn 


figEta = vis.Figure(title="Viscosity", quality=3)
figEta.append ( vis.objects.Points(swarmMaterials,fn_colour = fn_viscosity, fn_size=7 ))
#figEta.show() 


# Density & Buoyancy Functions
# --

# We scale compositional density using density changes driven by thermal expansion/contraction.  We then use a thermally driven Rayleigh number (Ra) and compositionally driven Rayleigh number (Rb) to build our buoyancy functions.  
# 
# $$
# \Delta\rho = \rho_{cl} - \rho_{m} = {B}(\alpha \rho_{m} \Delta T)
# $$
# 
# $$
# \rho_{cl}= {B}(\alpha \rho_{m} \Delta T) + \rho_{m}
# $$
# 
# $$
# Ra = \frac{\alpha\rho_{m} g \Delta T h^3}{\kappa \eta_{ref}}   ;   Rb = \frac{ \Delta\rho g h^3}{\kappa\eta_{ref}}   $$
# 
# where $\rho_{cl}$ is the average continental lithosphere, $\rho_{m}$ is a reference mantle density, $\alpha$ is thermal expansion coefficient, and $\Delta T$ is the temperature drop across the system.  B refers to the non-dimensional number (sometimes called the buoyancy ratio) that you use to scale to get the desired compositional density. When B is zero, this means that the material has the same density as the reference mantle.
# 


#density

BCont1 = -0.92
BMantle = 0.0

refDensMap = {       continent1Material: BCont1,
                     refmantleMaterial: BMantle}


# Rayleigh number.
Ra = 1.0e5  # make sure this matches what you used in your start-up models.  also, watch your resolution if you set this higher

Rb = 1.0e5  #NEEDS TO BE SAME AS RA sets up buoyancy scheme...again don't worry about this now, but if we want to impose different densities later we'll need it

# Define our vertical unit vector using a python tuple (this will be automatically converted to a function).
z_hat = ( 0.0, 1.0 )

contbuoy = True # set this to true if you plan on using different densities for the continental material

if contbuoy:
    # construct the density function using material properties outlined above
    densityFn = fn.branching.map( fn_key = materialIndex, mapping = refDensMap )
    # creating a buoyancy force vector
    buoyancyFn = (Ra * temperatureField - Rb * densityFn)  * z_hat
    
else:
    # Construct our density function.
    densityFn = Ra * temperatureField
    # Now create a buoyancy force vector using the density and the vertical unit vector. 
    buoyancyFn = densityFn * z_hat


if  writefigures:
    figtemp.save_image(outputPath +"TemperatureField_0000")
    figEta.save_image(outputPath +"Viscosity_0000")
    figMaterialMesh.save_image(outputPath +"Materials_0000")

# Output model timestep info    
    
start = timekeeper.time()

if rank == 0:
    fw = open(outputPath + "FrequentOutput.dat","w")
    fw.write("%s \n" %(timekeeper.ctime()))
    fw.close()
    


# System Setup
# -------
# **Setup a Stokes system**
# 
# Underworld uses the Stokes system to solve the incompressible Stokes equations.  

stokes = uw.systems.Stokes( velocityField = velocityField, 
                            pressureField = pressureField,
                            conditions    = velBC,
                            fn_viscosity  = fn_viscosity, 
                            fn_bodyforce  = buoyancyFn )

# get the default stokes equation solver
solver = uw.systems.Solver( stokes )


# **Set up the advective diffusive system**
# 
# Underworld uses the AdvectionDiffusion system to solve the temperature field given heat transport through the velocity field.



advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField, 
                                         fn_diffusivity = 1.0, 
                                         conditions     = tempBC )

# Create a system to advect the swarm YOU MUST USE THIS IF YOU USE SWARMS TO PUT IN MATERIALS
advector = uw.systems.SwarmAdvector( swarm=swarmMaterials, velocityField=velocityField, order=2 )

#Adding Advecting Tracers


tracerParticles1 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
advector_tracer1 = uw.systems.SwarmAdvector( swarm=tracerParticles1, velocityField=velocityField, order=2 )
tracerParticles2 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
advector_tracer2 = uw.systems.SwarmAdvector( swarm=tracerParticles2, velocityField=velocityField, order=2 )
tracerParticles3 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
advector_tracer3 = uw.systems.SwarmAdvector( swarm=tracerParticles3, velocityField=velocityField, order=2)

# Set up location of tracers

tracerCoords = np.array([[(cont1X1+(cont1X2-cont1X1)/2.0), 1.0],[(cont1X1+(cont1X2-cont1X1)/64.0), 1.0],[(cont1X1+63.0*(cont1X2-cont1X1)/64.0), 1.0]])
tracerParticles1.add_particles_with_coordinates(tracerCoords[0].reshape(1,2))
tracerParticles2.add_particles_with_coordinates(tracerCoords[1].reshape(1,2))
tracerParticles3.add_particles_with_coordinates(tracerCoords[2].reshape(1,2))

# set up sampling call

import mpi4py

def sampleTracers():
    arrY_temp = np.linspace(0.,1.,n)
    arrT1_temp = np.zeros(n)
    arrT2_temp = np.zeros(n)
    arrT3_temp = np.zeros(n)
    xpos1_temp = 0.
    xpos2_temp = 0. 
    xpos3_temp = 0.

    if len(tracerParticles1.data) > 0:
       xpos1_temp = tracerParticles1.data[0,0]
    if len(tracerParticles2.data) > 0:   
       xpos2_temp = tracerParticles2.data[0,0]
    if len(tracerParticles3.data) > 0:
       xpos3_temp = tracerParticles3.data[0,0]
    
    xpos1_global = uw.mpi.comm.allreduce(xpos1_temp, op=mpi4py.MPI.SUM)
    xpos2_global = uw.mpi.comm.allreduce(xpos2_temp, op=mpi4py.MPI.SUM)
    xpos3_global = uw.mpi.comm.allreduce(xpos3_temp, op=mpi4py.MPI.SUM)

    for i in range(n):
        arrT1_temp[i] = temperatureField.evaluate_global((xpos1_global,arrY_temp[i]))
        arrT2_temp[i] = temperatureField.evaluate_global((xpos2_global,arrY_temp[i]))
        arrT3_temp[i] = temperatureField.evaluate_global((xpos3_global,arrY_temp[i]))

    if rank == 0:
       fw = open( outputPath + "TemperatureTracer1.dat","a")
       fw.write("Step \t Tracer Location \t Temperature \t Depth \t \n")
       for i in range(n):
           fw.write("%.4f \t %.4f \t %.4f \t %.4f \t \n" %(step, xpos1_global, arrT1_temp[i], arrY_temp[i])) 
       fw.close()
       fw = open( outputPath + "TemperatureTracer2.dat","a")
       fw.write("Step \t Tracer Location \t Temperature \t Depth \t \n")
       for i in range(n):
           fw.write("%.4f \t %.4f \t %.4f \t %.4f \t \n" %(step, xpos2_global, arrT2_temp[i], arrY_temp[i])) 
       fw.close()
       fw = open( outputPath + "TemperatureTracer3.dat","a")
       fw.write("Step \t Tracer Location \t Temperature \t Depth \t \n")
       for i in range(n):
           fw.write("%.4f \t %.4f \t %.4f \t %.4f \t \n" %(step, xpos3_global, arrT3_temp[i], arrY_temp[i]))
       fw.close()


sampleTracers()

tracerPoints = vis.objects.Points(tracerParticles1, pointSize=10)
figTracer = vis.Figure(title="Tracers", quality=3)
figTracer.append( tracerPoints )
figTracer.append (vis.objects.Points(tracerParticles2, pointSize=10))
figTracer.append (vis.objects.Points(tracerParticles3, pointSize=10))
figTracer.append (vis.objects.Points(swarmMaterials, materialIndex, pointSize=3.,  colours='purple red green blue gray'))


if writefigures:
   figTracer.save_image(outputPath +"Tracers_0000")


# **Analysis Tools**
# 
# **Nusselt number**
# 
# The Nusselt number is the ratio between convective and conductive heat transfer
# 
# \\[
# Nu = -h \frac{ \int_0^l \partial_z T (x, z=h) dx}{ \int_0^l T (x, z=0) dx}
# \\]
# 
# **RMS velocity**
# 
# The root mean squared velocity is defined by intergrating over the entire simulation domain via
# 
# \\[
# \begin{aligned}
# v_{rms}  =  \sqrt{ \frac{ \int_V (\mathbf{v}.\mathbf{v}) dV } {\int_V dV} }
# \end{aligned}
# \\]
# 
# where $V$ denotes the volume of the box.
# 
# 


nuTop    = uw.utils.Integral( fn=temperatureField.fn_gradient[1], 
                              mesh=mesh, integrationType='Surface', 
                              surfaceIndexSet=mesh.specialSets["MaxJ_VertexSet"])

nuBottom = uw.utils.Integral( fn=temperatureField,               
                              mesh=mesh, integrationType='Surface', 
                              surfaceIndexSet=mesh.specialSets["MinJ_VertexSet"])


nu = - nuTop.evaluate()[0]/nuBottom.evaluate()[0]
#print('Nusselt number = {0:.6f}'.format(nu))

intVdotV = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )

vrms = math.sqrt( intVdotV.evaluate()[0]/ volume [0] )
#print('Initial vrms = {0:.3f}'.format(vrms))


# Main time stepping loop
# -----

steps_end = 10000 + step
checkpointstep = 500
simtime = start


# initalize values
vrms, nu = 0.0, 0.0
dt = min(advector.get_max_dt(), advDiff.get_max_dt())

if rank ==0:
        fw = open( outputPath + "FrequentOutput.dat","a")
        fw.write("Setup time: %.2f seconds\n" %(timekeeper.time() - start))
        fw.write("--------------------- \n")
        fw.write("Step \t Time \t Stopwatch \t Average Temperature \t Nusselt Number \t Vrms\n")
        start = timekeeper.time()
        fw.close()

start = timekeeper.time() # Setup clock to calculate simulation CPU time.

trackHF = True

if trackHF:    
    arrMeanTemp = numpy.zeros(steps_end+1)
    arrNu = numpy.zeros(steps_end+1)
    arrVrms = numpy.zeros (steps_end+1)
    

# perform timestepping

while step < steps_end:
    # Solve for the velocity field given the current temperature field.
    solver.solve()
    dt = min(advector.get_max_dt(), advDiff.get_max_dt())
    advector.integrate(dt)
    advDiff.integrate(dt)
    advector_tracer1.integrate(dt)        #added for advecting tracers
    advector_tracer2.integrate(dt)
    advector_tracer3.integrate(dt)
    simtime += dt
    time += dt
    step += 1
    avTemperature = temperatureFieldIntegral.evaluate()[0]/volume[0]
    vrms = math.sqrt( intVdotV.evaluate()[0] / volume[0])
    nu = - nuTop.evaluate()[0]/nuBottom.evaluate()[0]
            
    if trackHF: 
        if rank == 0:
            arrMeanTemp[step] = avTemperature
            arrNu[step] = nu
            arrVrms[step] = vrms

    if rank==0:
        fw = open( outputPath  + "FrequentOutput.dat","a") 
        fw.write("%i \t %.6f \t %.2f \t  %.5f \t %.5f \t %.5f \t\n" %(step, time, timekeeper.time() - start, avTemperature, nu, vrms))
        start = timekeeper.time()
        fw.close()
        
    if step % checkpointstep == 0.:
        MeshHand=mesh.save(outputPath + "mesh_%i.h5" %step)
        TempInfo=temperatureField.save(outputPath +"temperature_%i.h5" %step, MeshHand )
        SwarmInfo=swarmMaterials.save(outputPath +"swarm_%i.h5" %step) 
        SwarmVarInfo=materialIndex.save(outputPath + "materialIndex_%i.h5" %step)
        VelInfo=velocityField.save(outputPath + "velocityField_%i.h5" %step, MeshHand)
        PressInfo=pressureField.save(outputPath + "pressureField_%i.h5" %step, MeshHand)
        velocityField.xdmf (outputPath + "velocityField_%i.xdmf" %step, VelInfo,"Velocity", MeshHand, "Mesh")
        temperatureField.xdmf(outputPath + 'temperature_%i.xdmf' %step, TempInfo, "Temperature", MeshHand, "Mesh")
        pressureField.xdmf(outputPath + 'pressure_%i.xdmf' %step, PressInfo, "Pressure", MeshHand, "Mesh")
        materialIndex.xdmf(outputPath + 'materials_%i.xdmf' %step, SwarmVarInfo, "Materials", SwarmInfo, "Swarm")
        
    if step % checkpointstep == 0. and writefigures:
        figtemp.save_image(outputPath +"TemperatureField_%i" %step)
        figEta.save_image(outputPath +"Viscosity_%i" %step)
        figMaterialMesh.save_image(outputPath +"Materials_%i" %step)
        figTracer.save_image(outputPath +"Tracers_%i" %step)
        sampleTracers()
 
    if step % 10.0 == 0.0:
        population_control.repopulate()
        
