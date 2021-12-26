#Author: Valentin Nikandrov

#Import modules and libs
import numpy as np
import cv2 as cv

import dlib
from skimage import io
from scipy.spatial import distance

#Import scripts
import prepare

#Persons
persons = ['I dont now who is this']

def GetFaceDesc(frame, shapePredic, faceRecog, detector):
	detsShape = detector(frame, 1)

	#Get face shape from frame
	for k, d in enumerate(detsShape):
		shapeFrame = shapePredic(frame, d)

	#Check shape exists
	if shapeFrame != None:
		faceDescFrame = faceRecog.compute_face_descriptor(frame, shapeFrame)
		return faceDescFrame
	else: return False

def Main(persons):
	#Capture video from camera
	cap = cv.VideoCapture(0)

	#Init variables
	shapePredic = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
	faceRecog = dlib.face_recognition_model_v1('../dlib_face_recognition_resnet_model_v1.dat')
	detector = dlib.get_frontal_face_detector()

	#Init Persons
	descPersons = []

	for i in range(0, len(persons)):
		try:
			img = io.imread('../../storage/faces/' + str(i) + '_c.jpg')

			faceDescImg = GetFaceDesc(img, shapePredic, faceRecog, detector)

			if faceDescImg != False:
				descPersons.append(faceDescImg)
		except Exception as e:
			print('Еrror: ' + traceback.format_exc())
			continue

	#RealTime processing
	while True:
		_, frame = cap.read()

		#Crop frame
		cropFrame = frame[50 : 150 + 251, 150 : 250 + 251]

		#Compress frame
		compressedFrame = prepare.CompressFrame(cropFrame, 30)

		#Init detector
		detsFrame = detector(cropFrame, 1)

		try:
			#Did outline of face
			for det in detsFrame:
				cv.rectangle(cropFrame ,(det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)

			#Did central square
			cv.rectangle(frame , (150, 50), (500, 400), (0, 0, 255), 3)

			faceDescFrame = GetFaceDesc(compressedFrame, shapePredic, faceRecog, detector)

			#Check GetFaceDesc result
			if faceDescFrame != False:
				succ = False

				#Check coincidence				
				for i in range(0, len(descPersons)):
					a = distance.euclidean(faceDescFrame, descPersons[i])	
					if a < 0.6:
						succ = True
						print('Person identified: ' + persons[i])
						cv.putText(frame, 'Person: ' + persons[i], (20,20), cv.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
						break
				if not succ:
					cv.putText(frame, 'Person not identified', (20,20), cv.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
			else:
				print('Shape is null')
				cv.putText(frame, 'Face was no detected', (10,10), cv.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
			print(a)  
		except Exception as e:
			pass

		#Return result
		cv.imshow('Result', frame)

		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	#Close program
	cap.release()
	cv.destroyAllWindows()

Main(persons)
