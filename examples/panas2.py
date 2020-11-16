import time
import cv2
import math
#import adafruit_amg88xx
from Adafruit_AMG88xx import Adafruit_AMG88xx
import picamera
import picamera.array
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from firebase import firebase
import datetime

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

t = datetime.datetime.now()

try:
    firebase = firebase.FirebaseApplication('https://termalcam-d9c8e.firebaseio.com/')
except:
    print ("Koneksi Gagal")


try:
	sensor = Adafruit_AMG88xx()		
	# Waiting for sensor initialization
	time.sleep(.1)

	while True:		

		with picamera.PiCamera() as camera:
			camera.resolution = (320, 240)
			camera.capture('./tmp.jpg')
		
		max_temp = max(sensor.readPixels())
		
		img0 = cv2.imread('./tmp.jpg')
		img = img0[0:240,41:280]
		img = img[:, :, ::-1].copy()
		
		
		
		plt.subplot(1,2,1)
				
		pixels = sensor.readPixels()
		
		pixels = [map(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels]
		
		bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')
		
		fig = plt.imshow(bicubic, cmap="inferno", interpolation="bicubic")
		plt.colorbar()
		
		#nax = max_temp + 2
		varA = 0.022380841
		varb1 = -0.00166934
		varb2 = 1.173835088
		mxs = 38
		rums100 = varA + (varb1 * 100) + (varb2 * max_temp) + 2.503336
		#rums150 = varA + (varb1 * 150) + (varb2 * max_temp) + 3.17472
		#rums200 = varA + (varb1 * 200) + (varb2 * max_temp) + 3.844105
		#rums250 = varA + (varb1 * 250) + (varb2 * max_temp) + 4.514489
		
		face = cv2.CascadeClassifier('face-detect.xml')
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		wajah = face.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in wajah:
			if rums100 >= mxs:
				cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
				cv2.putText(img, str("%.2f" % rums100) +' Celcius', (x, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 2)
				try:
					result = firebase.post('termalcam', {'tingkat':str("Tidak Aman"), 'rTemp':str("%.2f" % rums100), 'aTemp':str(max_temp), 'time':str(t)})
				except:
					print ("Koneksi Gagal")
			else:
				cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
				cv2.putText(img, str("%.2f" % rums100) +' Celcius', (x, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
				try:
					result = firebase.post('termalcam', {'tingkat':str("Aman"), 'rTemp':str("%.2f" % rums100), 'aTemp':str(max_temp), 'time':str(t)})
				except:
					print ("Koneksi Gagal")

			cv_warna = img[y:y+h, x:x+w]
			cv_gray = gray[y:y+h, x:x+w]
			print(rums100)
			print(max_temp)
		
		plt.subplot(1,2,2)
		plt.imshow(img)
		#plt.text(0, -10, str("%.2f" % rums250) +' deg Celcius',size = 40, color = "red")
		#plt.text(0, -10, str(max_temp) +' deg Celcius',size = 40, color = "red")

		plt.draw()

		plt.pause(0.01)
		plt.clf()

except KeyboardInterrupt:
	print("done")
