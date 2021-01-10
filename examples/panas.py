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

MINTEMP = 26
MAXTEMP = 32

COLORDEPTH = 1024
points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]

height = 480
width = 480


#some utility functions
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


displayPixelWidth = width / 30
displayPixelHeight = height / 30


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

		plt.subplot(1,2,2)
		plt.imshow(img)
		plt.text(0, 0, str(max_temp) +'deg Celcius',size = 20, color = "red")

		plt.draw()

		plt.pause(0.01)
		plt.clf()

except KeyboardInterrupt:
	print("done")
