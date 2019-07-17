import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_l = []
x_t = []

for r, d, files in os.walk(image_dir):
	for f in files:
		if f.endswith("png") or f.endswith("jpg"):
			path = os.path.join(r,f)
			label = os.path.basename(r).replace(" ", "-").lower()
			# print(label,path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]

			pil_image = Image.open(path).convert("L")
			# image_array = np.array(pil_image, "uint8")
			image_array = np.int8(pil_image)
			# print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			# print(faces)
			# print(image_array)
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_t.append(roi)
				y_l.append(id_)

with open("labels.pickle",'wb') as f:
	pickle.dump(label_ids,f)

recognizer.train(x_t, np.array(y_l))
recognizer.save("trainner.yml")