#Author: Valentin Nikandrov

#Import modules and libs
import cv2 as cv

#Compress frame
def CompressFrame(frame, scale_percent):
	width = int(frame.shape[1] * scale_percent / 100)
	height = int(frame.shape[0] * scale_percent / 100)

	dim = (width, height)

	compressedFrame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
	return compressedFrame
