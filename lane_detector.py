import cv2
import numpy as np
import sys

class LaneDetector(object):
    def __init__(self, movieName):

        self.vc = cv2.VideoCapture(movieName)
        if not self.vc.isOpened():
            print "Error opening video file"
            sys.exit(0)

        # skipping the first few frames (to where the car is on the road)
        for ii in range(500):
            self.vc.grab()

        self.combined =  [] #Combined image
        self.rightlines = [] # Right lane lines
        self.leftlines = [] # Lef lane lines
        self.vps = [] # Vanishing point ycoords


    def getFrame(self):
        self.ret, self.img = self.vc.read()

    def execute(self):
        self.getFrame()
        imglist = self.split_image(self.img)
        self.chevp_algorithm(imglist)
        # self.find_avg_vp()

        return self.img

    def solve_equations(self, (m1, c1), (m2, c2)):
        coeff = np.array([[1, -m1], [1, -m2]])
        intercept = np.array([c1, c2])
        soln = np.linalg.solve(coeff, intercept)
        return soln

    # Split image into multiple parts to identify even the curved parts
    def split_image(self, img):
        height = img.shape[0]
        imglist = [img[height/2:height, :], img[height/4:height/2, :], img[height/8:height/4, :],
                    img[height/16:height/8, :], img[height/32:height/16, :], img[height/64:height/32, :]]

        return imglist

    def pre_processing(self, img):
        blurred_img = cv2.GaussianBlur(img, (3,3), 0)
        return blurred_img

    def chevp_algorithm(self, imglist):
        for i, img in enumerate(imglist):
            # Do a canny edge detection
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,100,200, L2gradient=True)
            cv2.imwrite("canny" + str(i) + ".jpg", edges)

            # Set threshold based on the height of the image
            threshold  = int(0.5 *img.shape[0])
            threshold = threshold if  threshold >= 25 else 25

            lines = cv2.HoughLines(edges,1,np.pi/180, threshold)
            leftlines = []
            rightlines = []
            if lines is not None:
                for rho,theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    c = np.tan(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 700*(-b))
                    y1 = int(y0 + 700*(a))
                    x2 = int(x0 - 700*(-b))
                    y2 = int(y0 - 700*(a))

                    angle = np.arctan2(y2-y1, x2-x1) * 180.0 / np.pi

                    x3 = int(x2 + 400 * np.cos(np.arctan2(y2-y1,x2-x1)))
                    y3 = int(y2 - 400 * np.sin(np.arctan2(y2-y1,x2-x1)))

                    # Ignore unwanted lines by filtering a set of angles.
                    if -20 < angle < 20 or angle == 90 or angle == 0 or angle == -90:
                        pass
                    else:
                        if angle > 0:
                            self.leftlines.append((-1/c , rho*1/b))
                        else:
                            self.rightlines.append((-1/c,  rho*1/b))

                        cv2.line(img,(x1,y1), (x2,y2), (0,0,255), 2)

    def find_avg_vp(self):
        for x in range(len(self.leftlines)):
            for y in range(len(self.rightlines)):
                soln = self.solve_equations(self.leftlines[x], self.rightlines[y])
                self.vps.append(int(soln[0]))

        if len(self.vps) > 0:
            ycoord = int(np.mean(self.vps))
            cv2.line(self.img, (0, ycoord), (self.img.shape[0], ycoord), (0, 255, 0), 2)
        else:
            cv2.line(self.img, (0, int(soln[0])), (self.img.shape[0], int(soln[0])), (0, 255, 0), 2)