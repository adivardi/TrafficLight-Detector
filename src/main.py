#!/usr/bin/env python
# coding: utf-8
# created by hevlhayt@foxmail.com 
# Date: 2016/1/15 
# Time: 19:20
#
import os
import cv2
import numpy as np


def detect(filepath, file):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(filepath+file)
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # colors range
    lower_red1 = np.array([0, 0, 0])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160, 0, 0])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    # lower_yellow = np.array([15,100,100])
    # upper_yellow = np.array([35,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])

    # mask color by range
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)

    size = img.shape
    # print size

    # hough circle detect
    # return: vector of circles. each circle is (x, y, radius, vote)
    # params:
    #  dp: inverse ratio of the accumulator resolution to the image resolution (see Yuen et al. for more details).
    #       Essentially, the larger the dp gets, the smaller the accumulator array gets.
    #  minDist: minimum distance between the centers of the detected circles. If the parameter is too small, multiple
    #           circles in the same neighborhood as the original may be (falsely) detected.
    #           If the minDist is too large, then some circles may not be detected at all.
    #  param1: First method-specific parameter. In case of HOUGH_GRADIENT and HOUGH_GRADIENT_ALT, it is the higher
    #          threshold of the two passed to the Canny edge detector(the lower one is twice smaller).
    #          Note that HOUGH_GRADIENT_ALT uses Scharr algorithm to compute image derivatives, so the threshold value
    #          should normally be higher, such as 300 or normally exposed and contrasty images.
    #  param2: Second method-specific parameter. In case of HOUGH_GRADIENT, it is the accumulator threshold for the
    #          circle centers at the detection stage. The smaller it is , the more false circles may be detected.
    #          Circles corresponding to the larger accumulator values, will be returned first.
    #          In the case of HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure.
    #          The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
    #          If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
    #          But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
    #  minRadius: Minimum size of the radius (in pixels).

    # It also helps to smooth image a bit unless it's already soft.
    # to reduce the noise in the image. In edge detection, numerical derivatives of the pixel intensities have to be
    # computed, and this typically results in ‘noisy’ edges
    # For example, GaussianBlur() with 7x7 kernel and 1.5x1.5 sigma or similar blurring may help

    maskr = cv2.GaussianBlur(maskr, (5, 5), 1)
    maskg = cv2.GaussianBlur(maskg, (5, 5), 1)
    masky = cv2.GaussianBlur(masky, (5, 5), 1)

    r_circles = cv2.HoughCircles(image=maskr, method=cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    g_circles = cv2.HoughCircles(image=maskg, method=cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    y_circles = cv2.HoughCircles(image=masky, method=cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    # traffic light detect
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            # discard circles based on some x/y limits ???
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            # count ratio of red pixels over total pixels inside circle
            for m in range(-r, r):
                for n in range(-r, r):

                    # discard circles near the image edge ??
                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            # if ratio red/total is high, keep
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]*2, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]*2, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]*2, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    cv2.imshow('img', img)
    # cv2.imshow('hsv', hsv)
    cv2.imshow('maskr', maskr)
    cv2.imshow('maskg', maskg)
    # cv2.imshow('masky', masky)

    cv2.imshow('detected results', cimg)
    # cv2.imwrite(path+'//result//'+file, cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # path = os.path.abspath('..')+'//light//'
    path = '/home/adi/Projects/traffic_lights/imgs/'
    # path = '/home/adi/Projects/traffic_lights/ws/src/TrafficLight-Detector/light/'
    for filename in os.listdir(path):
        print(f"path: {path}")
        print(f"file: {filename}")
        if filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.png') or filename.endswith('.PNG'):
            detect(path, filename)

