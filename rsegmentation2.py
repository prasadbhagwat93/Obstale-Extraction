import numpy as np, cv2

class RoadColorSegmentation(object):

    def __init__(self, movieName):
        self.vc = cv2.VideoCapture(movieName)
        if not self.vc.isOpened():
            print 'Error opening video file'
            sys.exit(0)

        # skipping the first few frames (to where the car is on the road)
        for ii in range(500):
            self.vc.grab()

        # reading and sub-sampling the image
        (ret, self.imgColor) = self.vc.read()
        self.subSampling = 1
        self.imgColor = self.imgColor[::self.subSampling, ::self.subSampling, :].copy()

        # size of the image
        self.m, self.n, o = np.shape(self.imgColor)

        self.distanceParams = [4, 0.02]

    def execute(self):
        self.runEstimation()
        return self.labels, self.imgColorLabeled


    def setupDistanceCost(self, ellipseRatio = 3, penalty = 0.02):
        xv = np.array(range(self.m), dtype = 'float32')
        xv[xv < self.m//3.5] = 0.
        yv = np.array(range(self.n), dtype = 'float32')
        xv = xv - self.x
        yv = yv - self.y
        self.distCost = [[penalty * (ellipseRatio**2 * xi**2 + yi**2)**0.5 for yi in yv] for xi in xv]
        self.distCost = np.array(self.distCost, dtype = 'float32')

    def getPixelMahalanobisDistance(self):
        img = np.reshape(self.imgColor.astype(np.float32), (self.m*self.n, 3))
        imgCentered = img - self.sampleMean
        temp = np.dot(imgCentered, self.sampleInf)
        distance = np.sum(temp * imgCentered, axis = 1)
        distance = np.reshape(distance, (self.m, self.n))

        return distance

    def colorParams(self):
        self.labels = np.zeros((self.m, self.n), dtype = 'uint8')
        samples = self.imgColor[self.x-2*self.wid:self.x+2*self.wid+1, self.y-self.wid:self.y+self.wid+1, :].astype(np.float32)
        samples = np.reshape(samples, ((4*self.wid + 1) * (2*self.wid+1), 3))

        # computing the mean and covariance of the RGB (BGR in OpenCV) values
        # of the samples
        self.sampleMean = np.mean(samples, axis = 0)
        self.sampleInf = np.linalg.inv(3 * np.cov(samples.T))


    def displayLabeled(self):

        # Giving a red tint to the pixels labeled as road.
        self.imgColorLabeled = self.imgColor.copy()
        self.imgColorLabeled[self.labels == 255, 2] = 255

        # Showing where the samples were taken from in the image.
        self.imgColorLabeled[self.x-2*self.wid : self.x+2*self.wid, self.y-self.wid: self.y+self.wid, 1] = 255


    #     cv2.imshow('Labels Colored', self.imgColorLabeled)
    #     cv2.imshow('Image', self.imgColor)
    #     cv2.imshow('Labels', self.labels)
    #     if runType == 1:
    #         # cv2.namedWindow('Alpha Image', cv2.CV_WINDOW_AUTOSIZE)
    #         cv2.imshow('Alpha Image', self.imgAlpha)

    #     cv2.waitKey(10)


    def closeFile(self):
        self.vc.release()


    def getFrame(self):
        retval, self.imgColor = self.vc.read()
        self.imgColor = self.imgColor[::self.subSampling, ::self.subSampling, :].copy()


    def setupRecursiveBayesianColorEstimator(self):

        self.prior = np.copy(self.distCost)


    def recursiveBayesionColorEstimator(self, thresh = 4.5):

        likelihood = self.getPixelMahalanobisDistance()

        likelihood += self.prior
        likelihood += self.distCost

        likelihood -= 35

        self.prior = likelihood
        self.labels[self.prior <  thresh] = 255

        # making sure the prior doesn't get too big, allowing pixels to become
        # raod more quickly
        self.prior[self.prior > 65.] = 65.


    def setUp(self, distParams = [], parameters = []):
        # types are
        # 2 - parameteric, recursive bayesian rgb-color based estimation

        # reading and sub-sampling the image
        # self.imgColor = self.imgColor
        self.subSampling = 1
        self.imgColor = self.imgColor[::self.subSampling, ::self.subSampling, :].copy()

        # size of the image
        self.m, self.n, o = np.shape(self.imgColor)
        # self.imgColor = img

        # point in image assumed to be part of the road
        self.x = self.m // 3.5 * 2
        self.y = self.n // 2
        self.wid = 10

        if len(distParams) > 0:
            self.setupDistanceCost(distParams[0], distParams[1])

        self.setupRecursiveBayesianColorEstimator()


    def runEstimation(self, parameters = []):
        self.getFrame()
        if len(parameters) > 0:
            self.colorParams()
            self.recursiveBayesionColorEstimator(parameters[0])
        else:
            self.colorParams()
            self.recursiveBayesionColorEstimator()


        self.displayLabeled()


# if __name__ == '__main__':
#     method = 2
#     # method = 1 still needs some work

#     distanceParams = [[4, 0.02], [4, 0.005], [4, 0.02]]
#     searchParams = [180] #only used when method = 1
#     filterParams = [[8], [1.0, 3.0], []]

#     RS = RoadColorSegmentation()

#     RS.setUp(method, distanceParams[method], searchParams)
#     for ii in range(2000):
#         RS.runEstimation(method, filterParams[method])

#     RS.closeFile()
