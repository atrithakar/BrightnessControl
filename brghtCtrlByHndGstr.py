'''
Project name: Brightness Control By Hand Gesture
Starting Date: 01 July 2023
Finishing Date: 03 July 2023
Author name: Thakar Atri Kamleshkumar
'''
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
import keyboard

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.7,max_num_hands=1)

Draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
b_per = 0
b_bar = 400
while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)

	cv2.putText(frame,f"Press 'S' key to save the brightness level",(40,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)
	cv2.putText(frame,f"Press 'Q' key to exit the code",(40,60),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)
	cv2.rectangle(frame,(50,150),(85,400),(255,0,0),3)
	cv2.rectangle(frame,(400,400),(600,450),(0,0,0),-1)
	cv2.putText(frame,f"(C) Atri Thaker",(405,430),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)

	frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	Process = hands.process(frameRGB)
	landmarkList = []
	if Process.multi_hand_landmarks:
		for handlm in Process.multi_hand_landmarks:
			for _id, landmarks in enumerate(handlm.landmark):
				height, width, color_channels = frame.shape
				x, y = int(landmarks.x*width), int(landmarks.y*height)
				landmarkList.append([_id, x, y])

	if landmarkList != []:
		x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
		x_2, y_2 = landmarkList[8][1], landmarkList[8][2]
		L = hypot(x_2-x_1, y_2-y_1)
		cv2.line(frame, (x_1, y_1), (x_2, y_2), (255, 0, 0), 2)
		cv2.circle(frame, (x_1, y_1), 15, (0, 255, 0), cv2.FILLED)
		cv2.circle(frame, (x_2, y_2), 15, (0, 255, 0), cv2.FILLED)
		cv2.circle(frame, ((x_1+x_2)//2, (y_1+y_2)//2), 15, (0, 0, 255), cv2.FILLED)
		if L<50:
			cv2.circle(frame, ((x_1+x_2)//2, (y_1+y_2)//2), 15, (0, 255, 255), cv2.FILLED)
		b_level = np.interp(L, [50, 200], [0, 100])
		b_per = np.interp(L, [50,200], [0,100])
		b_bar = np.interp(L, [50,200], [400,150])
		if keyboard.is_pressed('s'):
			sbc.set_brightness(int(b_level))
			print(f"Brightness level {int(b_level)} saved")

	cv2.rectangle(frame,(50,int(b_bar)),(85,400),(255,0,0),-1)
	cv2.putText(frame,f'{int(b_per)}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
	cv2.imshow('Image', frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()