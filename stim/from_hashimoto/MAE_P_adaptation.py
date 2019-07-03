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

vstimname = 'MAE_P_Aadaptation.py'

frameRate = 5.0 #(frames/sec)
base_sec=1
stim_sec_list=numpy.array([2,4,8,16,32])
post_sec=10
pre_stim=base_sec*frameRate
Nframes_stim_list=stim_sec_list*frameRate
post_stim = post_sec*frameRate
Ndir_per_trial = 2
Nduration_per_trial = 5
Nstim_per_trial=Ndir_per_trial*Nduration_per_trial
Ntrials=2
total_frames=((pre_stim+post_stim)*Nduration_per_trial + numpy.sum(Nframes_stim_list) )*Ndir_per_trial*Ntrials

nDots = 1500
SpeedDegPerSec = 20.0 # deg/sec
maxSpeed=SpeedDegPerSec/60.0

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
dir_deg=range(0.0-screen_tilt_deg,360.0-screen_tilt_deg,360.0/Ndir_per_trial)

#deg,dur = numpy.meshgrid(dir_deg,Nframes_stim_list)
dur,deg = numpy.meshgrid(Nframes_stim_list,dir_deg)
deg = numpy.ravel(deg)
dur = numpy.ravel(dur)
pre=numpy.tile(pre_stim,[1,Nstim_per_trial])[0,:]
post=numpy.tile(post_stim,[1,Nstim_per_trial])[0,:]

x_center = 0
y_center = 0

Device_Controler = "Dev2/ctr0"

flagRandomize = 0 # random stimulus order = 1
stim_log = vstim_utils.arrange_stim_order(Nstim_per_trial, Ntrials, flagRandomize)

LogSaveDirectory = 'C:\\Users\\mouse004\\Documents\\Research\\vstim\\psychopy\\MT_MST\\log'
log_hdr_name = 'MAE_P_A'

######################################################################################
## Save Vstim parameters before starting
######################################################################################
#save variables as mat file
TS = int32()
TS.nDots = nDots
TS.dir_deg=dir_deg
TS.Ndir_per_trial=Ndir_per_trial
TS.Nduration_per_trial=Nduration_per_trial
TS.Nstim_per_trial=Nstim_per_trial
TS.frameRate = frameRate
TS.stim_duration_sec_list = stim_sec_list
TS.base_duration_sec = base_sec
TS.post_duration_sec = post_sec
TS.Ntrials = Ntrials
TS.pre_stim = pre_stim
TS.Nframes_stim_list = Nframes_stim_list
TS.post_stim = post_stim
TS.total_frames = total_frames
TS.refreshRate = refreshRate
TS.stim_log = stim_log
TS.dotSize = dotSize
TS.maxSpeed = maxSpeed
TS.flagRandomize = flagRandomize
TS.vstimname = vstimname
TS.screen_tilt_deg = screen_tilt_deg
#TS.version_info = version_info
TS.x_center = x_center
TS.y_center = y_center
TS.deg = deg
TS.dur = dur

#vstim_utils.save_TS(LogSaveDirectory, log_hdr_name, TS)

######################################################################################
## Present Vstim
######################################################################################

#--------------------------------------------------------------------------------------------------------------------------------------

win = visual.Window([WinSizeX,WinSizeY], allowGUI=False, fullscr=False, screen=ScreenNumber, monitor = MonitorName ,rgb=[-1,-1,-1])
dots = visual.ElementArrayStim(win ,elementTex=None, fieldPos=(x_center,y_center), elementMask='circle' , nElements=nDots, units='deg' ,sizes=dotSize)

dotsTheta=numpy.random.rand(nDots)*360
dotsRadius=((numpy.random.rand(nDots)**0.5)*limit)
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

stim_log = numpy.array(stim_log,dtype='int8')
deg = deg[[stim_log]]
dur = dur[[stim_log]]
pre = pre[[stim_log]]
post = post[[stim_log]]

speedX = math.cos(deg[trial_counter]/180.0*math.pi)*maxSpeed
speedY = math.sin(deg[trial_counter]/180.0*math.pi)*maxSpeed

'''#---------------------------------------------------------------------fieldpositioncheck---------------------------------------------------------------------------------#
fixationmarker = visual.ElementArrayStim(win ,elementTex=None, fieldPos=(x_center,y_center), elementMask='circle' , nElements=1, units='deg' ,sizes=2,colors=[1,-1,-1])
fixationmarker.setXYs(numpy.array([numpy.array([x_center]),numpy.array([y_center])]).transpose())
fixationmarker.draw()
win.flip()
core.wait(10)
#---------------------------------------------------------------------fieldpositioncheck---------------------------------------------------------------------------------#'''

print stim_log

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
       #print trial_counter
    #---------comment out when using NIDAQ---------------------#'''

    if 0<data.value and data.value<numpy.sum(pre[0:trial_counter+1])+numpy.sum(dur[0:trial_counter])+numpy.sum(post[0:trial_counter])+1:
           dots.draw()
           win.flip()

    elif data.value > numpy.sum(pre[0:trial_counter+1])+numpy.sum(dur[0:trial_counter])+numpy.sum(post[0:trial_counter]) and data.value < numpy.sum(pre[0:trial_counter+1])+numpy.sum(dur[0:trial_counter+1])+numpy.sum(post[0:trial_counter])+1:
           frame = frame+1
           dotsX = dotsX+(speedX)
           dotsY = dotsY+(speedY)
           dots.setXYs(numpy.array([dotsX, dotsY]).transpose())
           dots.draw()
           win.flip()

           dotsTheta,dotsRadius = misc.cart2pol(dotsX, dotsY)
           outFieldDots = numpy.greater_equal(dotsRadius , limit)
           randomtheta = numpy.arccos((numpy.random.rand(sum(outFieldDots))*2.0-1.0))/math.pi*180
           dotsTheta[outFieldDots] = numpy.mod(randomtheta+deg[trial_counter]+90,360)
           dotsRadius[outFieldDots] = numpy.ones(sum(outFieldDots))*limit
           dotsX, dotsY = misc.pol2cart(dotsTheta,dotsRadius)

    else:
        dots.draw()
        win.flip()

    if data.value >= numpy.sum(pre[0:trial_counter+1])+numpy.sum(dur[0:trial_counter+1])+numpy.sum(post[0:trial_counter+1]): #data.value > (pre_stim+Nframes_stim+post_stim)*(trial_counter+1):
         trial_counter+=1
         trial_counter = numpy.min([trial_counter,trial_counter_max])
         frame=0
         speedX = math.cos(deg[trial_counter]/180.0*math.pi)*maxSpeed
         speedY = math.sin(deg[trial_counter]/180.0*math.pi)*maxSpeed

    for keys in event.getKeys():
         if keys in ['escape','q']:
            print myWin.fps()
            myWin.close()
            trial_counter=Nstim_per_trial*Nrep+1

''#---------comment out when not using NIDAQ---------------------#
if taskHandle.value != 0:
        nidaq.DAQmxStopTask(taskHandle)
        nidaq.DAQmxClearTask(taskHandle)
        print "End of program, press Enter key to quit "
#---------comment out when not using NIDAQ---------------------#'''

win.close()
print "End of program, press Enter key to quit "
