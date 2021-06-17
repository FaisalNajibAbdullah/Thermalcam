# USAGE
# python detect_mask_video.py
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import time
import cv2
import os
import picamera
import picamera.array
import math
from Adafruit_AMG88xx import Adafruit_AMG88xx
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from firebase import firebase
import datetime 
import RPi.GPIO as GPIO
import time

t = datetime.datetime.now()

try:
    firebase = firebase.FirebaseApplication('https://termalcam-d9c8e.firebaseio.com/')
except:
    print ("Koneksi Gagal")

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

#---------------------------------------
plt.ion()
plt.subplots(figsize=(8, 4))



#low range of the sensor (this will be blue on the screen)
MINTEMP = 26

#high range of the sensor (this will be red on the screen)
MAXTEMP = 32

#how many color values we can have
COLORDEPTH = 1024
points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]

#sensor is an 8x8 grid so lets do a square
height = 480
width = 480

#some utility functions
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

displayPixelWidth = width / 30
displayPixelHeight = height / 30
#---------------------------------------

#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
 
#set GPIO Pins
GPIO_TRIGGER = 18
GPIO_ECHO = 24
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
 
def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance
#---------------------------------------

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)

# loop over the frames from the video stream
try:
	sensor = Adafruit_AMG88xx()		
	# Waiting for sensor initialization
	time.sleep(.1)
	
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		with picamera.PiCamera() as camera:
				camera.resolution = (320, 240)
				camera.capture('./tmp.jpg')
				
		max_temp = max(sensor.readPixels())
		
		frame0 = cv2.imread('./tmp.jpg')
		frame = frame0[0:240,41:280]
		#frame = cv2.bilateralFilter(frame,5,25,25)
		frame = frame[:, :, ::-1].copy()
		
		plt.subplot(1,2,1)
				
		pixels = sensor.readPixels()
		
		pixels = [map(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels]
		
		bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')
		
		fig = plt.imshow(bicubic, cmap="inferno", interpolation="bicubic")
		plt.colorbar()

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		
		dist = distance()
		varA = 26.83741554
		varb1 = -0.00966282
		varb2 = 0.324591334
		mxs = 38
		rums100 = varA + (varb1 * 100) + (varb2 * max_temp) + 2.503336

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			#color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

			if dist <= 49 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 1
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 49 and dist <= 99 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 1.5
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 99 and dist <= 149 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 2
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 149 and dist <= 169 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 2.5
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 169 and dist <= 199 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 2.7
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 199 and dist <= 219 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 3.2
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 219 and dist <= 249 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 3.5
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 249 and dist <= 269 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 3.7
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 269 and dist <= 300 :
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 4
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			elif dist > 300:
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 5
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"
			else:
				rums = varA + (varb1 * dist) + (varb2 * max_temp) + 5
				label2 = label + str(" : " + "%.2f" % rums + " C")
				if rums < 38 and label == "Mask":
					color = (0, 255, 0)
					aman = "Suhu Normal"
				elif rums < 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				elif rums >= 38 and label == "No Mask":
					color = (255, 0, 0)
					aman = "Suhu Tinggi"
				elif rums >= 38 and label == "Mask":
					color = (255, 0, 0)
					aman = "Suhu Normal"
				else:
					color = (255, 0, 0)
					aman = "Suhu Normal"

			# akurasi suhu
			# label = label + str(" : " + "%.2f" % rums100 + " C")
			
			# akurasi masker
			#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			try:
				result = firebase.post('termalcam', {'tingkat':str("aman"), 'rTemp':str("%.2f" % rums), 'aTemp':str(max_temp), 'jarak':str("%.1f" % dist), 'mask':str(label), 'time':str(t)})
			except:
				print ("Koneksi Gagal")
				
			print(max_temp)
			print(rums)
			print("%.1f " % dist)
			print(label)
			print(aman)

		plt.subplot(1,2,2)
		plt.imshow(frame)

		plt.draw()

		plt.pause(0.01)
		plt.clf()

except KeyboardInterrupt:
	print("done")

