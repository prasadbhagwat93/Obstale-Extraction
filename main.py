import cv2 
import numpy as np
import lane_detector as ld
import road_segmentation as rs
import rsegmentation2 as rs2
import sys
import cv2utils

if __name__ == '__main__':

    if len(sys.argv) > 1:
        movieName = sys.argv[1]
        vc = cv2.VideoCapture(movieName)
        if not vc.isOpened():
          print "Error opening video file"
          sys.exit(0)
    else:
        print "Please provide input file"
        sys.exit(0)

    if len(sys.argv) > 2:
      method = int(sys.argv[2])
      print method
      if method != 1 and method != 2:
        print "Invalid method number"
    else:
      method = 1

    for ii in range(500):
      vc.grab()

    # Lane detection
    LD = ld.LaneDetector(movieName)
    RS = rs.RoadSegmentation(movieName)
    RS2 = rs2.RoadColorSegmentation(movieName)
    RS2.setUp(RS2.distanceParams)
    BM = rs.BModeller(movieName)

    while(1):
      re, frame = vc.read()

      lane_frame = LD.execute()
      cv2.imshow('Lane detection', lane_frame)

      if method == 1:
        road_frame = RS.execute()
        road_frame_inv = cv2utils.invert_mask(road_frame)
        road_frame = road_frame_inv
      else:
        road_frame, road_frame_color = RS2.execute()
        road_frame_inv = cv2utils.invert_mask(road_frame)
        cv2.imshow('Road segmentation colored', road_frame_color)
        road_frame = road_frame_inv

      cv2.imshow('Road Segmentation', road_frame)
      bgfg_frame = BM.execute()
      cv2.imshow('Background Modelling', bgfg_frame)

      # Combining two masks
      temp = cv2utils.apply_mask(road_frame, bgfg_frame)
      obstacle_extraction = cv2utils.apply_mask(frame, temp)

      gray_image = cv2.cvtColor(obstacle_extraction, cv2.COLOR_BGR2GRAY)
      contours,hierarchy = cv2.findContours(gray_image ,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

      for cnt in contours:
          if cv2.contourArea(cnt) > 200:
              [x,y,w,h] = cv2.boundingRect(cnt)
              if  h > 30 and w/h > 0.8:
                  cv2.rectangle(obstacle_extraction,(x,y),(x+w,y+h),(0,0,255),2)
              else:
                  cv2.rectangle(obstacle_extraction,(x,y), (x+w, y+h), (0, 255, 0), 2)

      cv2.imshow('Obstacle extraction', obstacle_extraction)

      k = cv2.waitKey(50) & 0xFF
      if k == 27:
          break

    cv2.destroyAllWindows()



