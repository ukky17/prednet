#!/usr/bin/env python2

import numpy as np
import math
import cv2
import hickle

def cart2pol(x, y, units='deg'):
    """Convert from cartesian to polar coordinates.
    :usage:
        theta, radius = cart2pol(x, y, units='deg')
    units refers to the units (rad or deg) for theta that should be returned
    """
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    if units in ('deg', 'degs'):
        theta = theta * 180 / np.pi
    return theta, radius

def pol2cart(theta, radius, units='deg'):
    """Convert from polar to cartesian coordinates.
    usage::
        x,y = pol2cart(theta, radius, units='deg')
    """
    if units in ('deg', 'degs'):
        theta = theta * np.pi / 180.0
    xx = radius * np.cos(theta)
    yy = radius * np.sin(theta)

    return xx, yy

def create_circles(dotsX, dotsY):
    field = np.full(winsize, 0, dtype=np.uint8)
    for i in range(nDots):
        cv2.circle(field, (winsize[1] // 2 + int(dotsY[i]),
                           winsize[0] // 2 + int(dotsX[i])),
                   dotSize, (255, 255, 255), thickness=-1)
    return field

# stim parameters ------------------------------------------------------------
n_movies = 10

pre_frames = 0
stim_frames = 5
post_frames = 5

stimtype = 'in' # out or in

nDots = 2000
speedR = 0.1
speedO = 1 + speedR
speedI = 1 - speedR
dotSize = 15
winsize = (1280, 1600)
limit = int(np.sqrt(winsize[0] ** 2 + winsize[1] ** 2))

output_size = (128, 160)

############################ present stims  ##################################
total_frames = pre_frames + stim_frames + post_frames

filename_base = 'stims/OF_R_' + stimtype
movie_filename = filename_base + '.mp4'
img_filename = filename_base + '.hkl'
source_filename = filename_base + '_source.hkl'

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(movie_filename, fourcc, 30, (winsize[1], winsize[0]),
                         False)

img_all = np.zeros((n_movies * total_frames, ) + output_size +  (3, ))
source = []
for m in range(n_movies):
    dotsTheta = np.random.rand(nDots) * 360
    dotsRadius = (np.random.rand(nDots) ** 0.5) * limit
    dotsLimitIn = dotsRadius * (np.random.rand(nDots) ** 0.5)
    dotsX, dotsY = pol2cart(dotsTheta, dotsRadius)

    for f in range(total_frames):
        if f < pre_frames:
            field = create_circles(dotsX, dotsY)

        elif f < pre_frames + stim_frames:
            if stimtype == 'out':

                dotsRadius *= speedO
                outDots = np.greater_equal(dotsRadius, limit)
                dotsRadius[outDots] = np.random.rand(sum(outDots)) ** 0.5 * limit
                dotsTheta[outDots] = np.random.rand(sum(outDots)) * 360

            elif stimtype == 'in':
                dotsRadius *= speedI
                inDots = np.less_equal(dotsRadius, dotsLimitIn)
                dotsRadius[inDots] = limit
                dotsLimitIn[inDots] = np.random.rand(sum(inDots)) ** 0.5 * limit
                dotsTheta[inDots] = np.random.rand(sum(inDots)) * 360

            dotsX, dotsY = pol2cart(dotsTheta, dotsRadius)
            field = create_circles(dotsX, dotsY)

        else:
            field = create_circles(dotsX, dotsY)

        # save the first movie
        if m == 0:
            writer.write(field.astype('uint8'))

        # save
        for ch in range(3):
            img_all[m * n_movies + f, :, :, ch] = cv2.resize(field,
                                              (output_size[1], output_size[0]))
        source.append([str(m)] * total_frames)

    if m == 0:
        writer.release()

hickle.dump(img_all, img_filename, mode='w')
hickle.dump(source, source_filename, mode='w')
