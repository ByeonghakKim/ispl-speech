import numpy as np
import cv2

capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_alt2.xml')
mouse_cascade = cv2.CascadeClassifier('./haar/Mouth.xml')

while True :
	ret, frame = capture.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	mouses = mouse_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces :
		cv2.rectangle(frame, (x,y), (x+h, y+w), (255,0,0), 2)
	
	for (x, y, w, h) in mouses :
		cv2.rectangle(frame, (x,y), (x+h, y+w), (0, 255, 0), 2)
 
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q') : 
		print(0)
		break
		
capture.release()
cv2.destroyAllWindows()