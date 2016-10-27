import cv2
import numpy as np
import sys

class BModeller(object):
    def __init__(self, movieName):

        self.vc = cv2.VideoCapture(movieName)
        if not self.vc.isOpened():
            print "Error opening video file"
            sys.exit(0)

        # skipping the first few frames (to where the car is on the road)
        for ii in range(500):
            self.vc.grab()

        self.history = 10
        self.varThreshold = 16

        self.fgbg = cv2.BackgroundSubtractorMOG2(history = self.history, varThreshold = self.varThreshold)
        self.fgbg.setBool('detectShadows', False)


    def getFrame(self):
        self.ret, self.img = self.vc.read()

    def execute(self):
        self.getFrame()

        preprocessed_img = self.pre_processing(self.img)
        fgmask = self.fgbg.apply(preprocessed_img, learningRate=0.005)
        # post_processed = self.pre_processing(fgmask)
        return fgmask


    def pre_processing(self, img):
        # Apply gaussian blur to the image to reduce noise
        blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
        return blurred_img


class RoadSegmentation(object):
    def __init__(self, movieName):
        self.vc = cv2.VideoCapture(movieName)
        if not self.vc.isOpened():
            print "Error opening video file"
            sys.exit(0)

        # skipping the first few frames (to where the car is on the road)
        for ii in range(500):
            self.vc.grab()

    def getFrame(self):
        self.ret, self.img = self.vc.read()

    def execute(self):
        self.getFrame()
        preprocessed_img = self.pre_processing(self.img)
        th_image = self.otsu_thresholding(preprocessed_img)
        return th_image

    def pre_processing(self, img):
        # Apply gaussian blur
        blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
        # Convert colorspace to reduce color information to eliminate noises/shadows
        conv_frame = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(conv_frame)

        return s

    def otsu_thresholding(self, bin_img):
        ret, th_img = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th_img



