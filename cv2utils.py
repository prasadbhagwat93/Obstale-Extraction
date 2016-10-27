import cv2

def apply_mask(frame, mask):
	res = cv2.bitwise_and(frame, frame, mask=mask)
	return res

def erode_and_dilate():
	pass

def invert_mask(mask):
	res = cv2.bitwise_not(mask)
	return res

