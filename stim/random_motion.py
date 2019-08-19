#!/usr/bin/env python2

import numpy as np
import math
import cv2
import hickle

def create_circles(dotsX, dotsY):
    field = np.full(winsize, 0, dtype=np.uint8)
    for i in range(nDots):
        cv2.circle(field, (dotsX, dotsY),
                   dotSize, (255, 255, 255), thickness=-1)
    return field

# stim parameters ------------------------------------------------------------
n_movies = 10

pre_frames = 0
stim_frames = 50
post_frames = 0

nDots = 1 # 2000, 2400, 1300
maxSpeed = 40 # pixel/frame
dotSize = 30
winsize = (1920, 2240) # (1280, 1600), (1280, 1920), (810, 1620)
limit = int(np.sqrt(winsize[0] ** 2 + winsize[1] ** 2))

output_size = (192, 224) # (128, 160), (128, 192), (81, 162)

############################ present stims  ##################################
total_frames = pre_frames + stim_frames + post_frames

filename_base = str(total_frames) + 'frames'
if output_size == (128, 160):
    filename_base += '/random_motion'
else:
    filename_base += str(output_size[0]) + 'x' + str(output_size[1]) + '/random_motion'
movie_filename = filename_base + '.mp4'
img_filename = filename_base + '.hkl'
source_filename = filename_base + '_source.hkl'
speed_filename = filename_base + '_speed.hkl'

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(movie_filename, fourcc, 30,
                         (winsize[1], winsize[0]), False)

img_all = np.zeros((n_movies * total_frames, ) + output_size +  (3, ))
source = []
speed_all = np.zeros((n_movies, total_frames))
for m in range(n_movies):
    dotsTheta = np.random.rand(nDots) * 360
    dotsRadius = (np.random.rand(nDots) ** 0.5) * limit
    dotsX = winsize[1] // 2
    dotsY = winsize[0] // 2

    for f in range(total_frames):
        if f < pre_frames:
            field = create_circles(dotsX, dotsY)

        elif f < pre_frames + stim_frames:
            speed = int((np.random.rand() * 2 - 1) * maxSpeed)
            if dotsX + speed > winsize[1] - dotSize:
                speed = winsize[1] - dotSize - dotsX
            elif dotsX + speed < dotSize:
                speed = dotSize - dotsX

            field = create_circles(dotsX, dotsY)
            speed_all[m, f] = speed
            dotsX += speed

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
hickle.dump(speed_all, speed_filename, mode='w')
