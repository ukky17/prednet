#!/usr/bin/env python2

"""This demo requires a graphics card that supports OpenGL2 extensions. 
It shows how to manipulate an arbitrary set of elements using numpy arrays
and avoiding for loops in your code for optimised performance.
see also the elementArrayStim demo

"""
from psychopy import visual, logging, core, event, misc
#from psychopy.tools.coordinatetools import pol2cart
import numpy
import ctypes
from scipy import io
import os
import matplotlib
import pylab
nidaq = ctypes.windll.nicaiu # load the DLL #comment out when not using NIDAQ
import math
import vstim_utils

# Setup some typedefs and constants
# to correspond with values in
# C:\Program Files\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include\NIDAQmx.h
# the typedefs
int32 = ctypes.c_long
uInt32 = ctypes.c_ulong
uInt64 = ctypes.c_ulonglong
float64 = ctypes.c_double
TaskHandle = uInt32

#set parameter ------------------------------------------------------------------------------------------------------------------

vstimname = 'OF_R.py'

frameRate = 5.0 #(frames/sec)
base_sec=1
stim_sec=4
post_sec=1
pre_stim=base_sec*frameRate
Nframes_stim=stim_sec*frameRate
post_stim = post_sec*frameRate
Nstim_per_trial=2
Ntrials=10
total_frames=(pre_stim+Nframes_stim+post_stim)*Nstim_per_trial*Ntrials

nDots = 1000
speedR = 0.02
speedO = 1+speedR
speedI = 1-speedR
dotSize = 0.5

limit=40.0
#MonitorName='samsung_marmoset'
#MonitorName='samsung'
MonitorName='testMonitor'
WinSizeX = 1600
WinSizeY = 900
ScreenNumber = 1
screen_tilt_deg=0
refreshRate = 60

x_center = 0
y_center = 0

Device_Controler = "Dev2/ctr0"

flagRandomize = 1 # random stimulus order = 1
stim_log = vstim_utils.arrange_stim_order(Nstim_per_trial, Ntrials, flagRandomize)
#stim_log = [1,0,1,0,1,0,1,0]
print stim_log
LogSaveDirectory = 'C:\\Users\\mouse004\\Documents\\Research\\vstim\\psychopy\\MT_MST\\log'
log_hdr_name = 'OF_R_001'

######################################################################################
## Save Vstim parameters before starting
######################################################################################
#save variables as mat file
TS = int32()
TS.nDots = nDots
TS.Nstim_per_trial=Nstim_per_trial
TS.frameRate = frameRate
TS.stim_duration_sec = stim_sec
TS.base_duration_sec = base_sec
TS.post_duration_sec = post_sec
TS.Ntrials = Ntrials
TS.pre_stim = pre_stim
TS.Nframes_stim = Nframes_stim
TS.post_stim = post_stim
TS.total_frames = total_frames
TS.refreshRate = refreshRate
#TS.invrefreshRate = invrefreshRate
TS.stim_log = stim_log
TS.dotSize = dotSize
TS.speedO = speedO
TS.speedI = speedI
TS.flagRandomize = flagRandomize
#TS.stimtype = stimtype
TS.vstimname = vstimname
TS.screen_tilt_deg = screen_tilt_deg
#TS.version_info = version_info
TS.x_center = x_center
TS.y_center = y_center

#vstim_utils.save_TS(LogSaveDirectory, log_hdr_name, TS)

######################################################################################
## Present Vstim
######################################################################################

#--------------------------------------------------------------------------------------------------------------------------------------

win = visual.Window([WinSizeX,WinSizeY], allowGUI=False, fullscr=False, screen=ScreenNumber, monitor = MonitorName ,rgb=[-1,-1,-1])
dots = visual.ElementArrayStim(win ,elementTex=None, fieldPos=(x_center,y_center), elementMask='circle' , nElements=nDots, units='deg' ,sizes=dotSize)

dotsTheta=numpy.random.rand(nDots)*360
dotsLimit = numpy.ones(nDots)*limit
dotsRadius=(numpy.random.rand(nDots)**0.5)*limit
dotsLimitIn = dotsRadius*(numpy.random.rand(nDots)**0.5)
dotsX, dotsY = misc.pol2cart(dotsTheta,dotsRadius)
dots.setXYs(numpy.array([dotsX, dotsY]).transpose())

''#---------comment out when not using NIDAQ---------------------#

DAQmx_Val_Cfg_Default = int32(-1)
DAQmx_Val_Volts = 10348
DAQmx_Val_Rising = 10280
DAQmx_Val_Falling = 10171
DAQmx_Val_FiniteSamps = 10178
DAQmx_Val_GroupByChannel = 0
DAQmx_Val_CountUp=10128

def CHK(err):
    """a simple error checking routine"""
    err = 0;
    if err < 0:
        buf_size = 100
        buf = ctypes.create_string_buffer('\000' * buf_size)
        nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
        raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.value)))

# set channel
taskHandle = TaskHandle(0)
CHK(nidaq.DAQmxCreateTask("",ctypes.byref(taskHandle)))
CHK(nidaq.DAQmxCreateCICountEdgesChan(taskHandle,Device_Controler,"",DAQmx_Val_Falling,0,DAQmx_Val_CountUp))

# initialize trial and frame counters
CHK(nidaq.DAQmxStartTask(taskHandle))

#---------comment out when not using NIDAQ---------------------#'''

data = uInt32()
trial_counter=0
trial_counter_max = Nstim_per_trial*Ntrials-1
frame_counter=0
frame=0
screenflips=0
sequence = 0

while frame_counter < total_frames:
    '''#---------comment out when not using NIDAQ---------------------#
    CHK(nidaq.DAQmxReadCounterScalarU32(taskHandle,float64(10.0), ctypes.byref(data), None))
    if data.value>frame_counter:
        #print "counter %d " %(data.value)
        frame_counter=data.value
    #---------comment out when not using NIDAQ---------------------#'''

    ''#---------comment out when using NIDAQ----------------------    
    screenflips += 1
    if screenflips%(refreshRate / frameRate)==0:
       frame_counter += 1
       data.value = frame_counter
       #print frame_counter
       #print [frame_counter,trial_counter]
    #---------comment out when using NIDAQ---------------------#'''

    if 0<data.value and data.value<pre_stim+(pre_stim+Nframes_stim+post_stim)*trial_counter+1:
        dots.draw()
        win.flip()
         
    elif data.value > pre_stim+(pre_stim+Nframes_stim+post_stim)*trial_counter and data.value < 1+(pre_stim+Nframes_stim+post_stim)*(trial_counter+1)-post_stim:
        frame = frame+1
        if stim_log [trial_counter]==0:
            
            dotsRadius = (dotsRadius*speedO)
            outFieldDots = numpy.greater_equal(dotsRadius , limit)
            dotsRadius[outFieldDots] = numpy.random.rand(sum(outFieldDots))**0.5*limit
            dotsLimitIn[outFieldDots] = dotsRadius[outFieldDots]
            dotsTheta[outFieldDots] = numpy.random.rand(sum(outFieldDots))*360
           
        elif stim_log [trial_counter]==1:
            
            dotsRadius = (dotsRadius*speedI)
            inFieldDots = numpy.less_equal(dotsRadius , dotsLimitIn)
            dotsRadius[inFieldDots] = dotsLimit[inFieldDots]
            dotsLimitIn[inFieldDots] = numpy.random.rand(sum(inFieldDots))**0.5*limit
            dotsTheta[inFieldDots] = numpy.random.rand(sum(inFieldDots))*360
            
        dotsX, dotsY = misc.pol2cart(dotsTheta,dotsRadius)
        dots.setXYs(numpy.array([dotsX, dotsY]).transpose())
        dots.draw()
        win.flip()
            
            
    else:
        dots.draw()
        win.flip()
        
    if data.value >= (pre_stim+Nframes_stim+post_stim)*(trial_counter+1): #data.value > (pre_stim+Nframes_stim+post_stim)*(trial_counter+1):
         trial_counter+=1
         trial_counter = numpy.min([trial_counter,trial_counter_max])
         frame=0

    for keys in event.getKeys():
         if keys in ['escape','q']:
            print myWin.fps()
            myWin.close()
            trial_counter=Nstim_per_trial*Ntrials+1

''#---------comment out when not using NIDAQ---------------------#
if taskHandle.value != 0:
        nidaq.DAQmxStopTask(taskHandle)
        nidaq.DAQmxClearTask(taskHandle)
        print "End of program, press Enter key to quit "
#---------comment out when not using NIDAQ---------------------#'''

win.close()
print "End of program, press Enter key to quit "

