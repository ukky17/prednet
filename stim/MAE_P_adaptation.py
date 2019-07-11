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

pre_frames = 10
stim_frames = 20 # 2, 4, 8, 16, 32 sec
post_frames = 20
deg = 0 # 0, 180

nDots = 1300 # 2000, 2400, 1300
maxSpeed = 40 # pixel/frame
dotSize = 15
winsize = (810, 1620) # (1280, 1600), (1280, 1920), (810, 1620)
limit = int(np.sqrt(winsize[0] ** 2 + winsize[1] ** 2))

output_size = (81, 162) # (128, 160), (128, 192), (81, 162)

############################ present stims  ##################################
total_frames = pre_frames + stim_frames + post_frames

filename_base = str(total_frames) + 'frames'
if output_size == (128, 160):
    filename_base += '/MAE_P_deg' + str(deg)
else:
    filename_base += str(output_size[0]) + 'x' + str(output_size[1]) + '/MAE_P_deg' + str(deg)
movie_filename = filename_base + '.mp4'
img_filename = filename_base + '.hkl'
source_filename = filename_base + '_source.hkl'

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(movie_filename, fourcc, 30,
                         (winsize[1], winsize[0]), False)

img_all = np.zeros((n_movies * total_frames, ) + output_size +  (3, ))
source = []
for m in range(n_movies):
    dotsTheta = np.random.rand(nDots) * 360
    dotsRadius = (np.random.rand(nDots) ** 0.5) * limit
    dotsX, dotsY = pol2cart(dotsTheta, dotsRadius)

    speedX = math.cos((deg + 90) / 180.0 * math.pi) * maxSpeed
    speedY = math.sin((deg + 90) / 180.0 * math.pi) * maxSpeed

    for f in range(total_frames):
        if f < pre_frames:
            field = create_circles(dotsX, dotsY)

        elif f < pre_frames + stim_frames:
            dotsX += speedX
            dotsY += speedY

            field = create_circles(dotsX, dotsY)

            dotsTheta, dotsRadius = cart2pol(dotsX, dotsY)
            outDots = np.greater_equal(dotsRadius, limit)
            randomtheta = np.arccos((np.random.rand(sum(outDots)) * 2.0 - 1)) \
                          / math.pi * 180
            dotsTheta[outDots] = np.mod(randomtheta + deg + 180, 360)
            dotsRadius[outDots] = np.ones(sum(outDots)) * limit
            dotsX, dotsY = pol2cart(dotsTheta, dotsRadius)

        else:
            field = create_circles(dotsX, dotsY)

        # save the first movie
        if m == 0:
            writer.write(field.astype('uint8'))

        # save
        for ch in range(3):
            img_all[m * total_frames + f, :, :, ch] = cv2.resize(field,
                                              (output_size[1], output_size[0]))
        source.append(str(m))

    if m == 0:
        writer.release()

hickle.dump(img_all, img_filename, mode='w')
hickle.dump(source, source_filename, mode='w')
