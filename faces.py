import cv2
import os
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

cap = cv2.VideoCapture(0)


def grava_face():
	name = input("Nome:\n")

	try:
	    # Create target Directory
	    os.mkdir("images/"+name)
	    print("Directory " , name ,  " Created ") 
	except FileExistsError:
	    print("Directory " , name ,  " already exists")

	cont = 0
	while(True):
	
#		print(cont)
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		# name = 'images'
		
		for (x, y, w, h) in faces:
			# print(x,y,w,h)
			roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
			roi_color = frame[y:y+h, x:x+w]
			
			img_item = 'images/'+ name + "/" + name + "_" +str(cont)+".png"
			cv2.imwrite(img_item, roi_color)
			# cv2.imwrite(img_item, roi_gray)
			color = (255,0,0)
			stroke = 2
			width = x + w
			heigh = y + h
			cont += 1
			cv2.rectangle(frame, (x,y), (width, heigh), color, stroke)

		if cont == 100:
			break

def treina(image_dir, face_cascade, recognizer):
	print('Treinando Faces...')
	current_id = 0
	label_ids = {}
	y_l = []
	x_t = []

	for r, d, files in os.walk(image_dir):
		print('...')
		for f in files:
			if f.endswith("png") or f.endswith("jpg"):
				path = os.path.join(r,f)
				label = os.path.basename(r).replace(" ", "-").lower()
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1

				id_ = label_ids[label]

				pil_image = Image.open(path).convert("L")
				size = (550, 550)
				final_image = pil_image.resize(size, Image.ANTIALIAS)
				image_array = np.array(final_image, dtype="uint8")
				# image_array = np.array(pil_image, dtype="uint8")
				faces = face_cascade.detectMultiScale(image_array)

				for (x,y,w,h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_t.append(roi)
					y_l.append(id_)

	with open("labels.pickle",'wb') as f:
		pickle.dump(label_ids,f)

	recognizer.train(x_t, np.array(y_l))
	recognizer.save("trainner.yml")

def reconhece():
	print('Reconhecendo Faces...')
	recognizer.read("trainner.yml")

	labels = {"person_name":1}
	with open("labels.pickle",'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}
		
		
	cont = 0
	while(True):
		# Capture frame-by-frame
#		print(cont)
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		name = ''
		
		for (x, y, w, h) in faces:
			roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
			roi_color = frame[y:y+h, x:x+w]

			id_, conf = recognizer.predict(roi_gray)
			if conf >= 45:
				# print(id_)
				# print(labels[id_])
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255,255,255)
				stroke = 2
				cv2.putText(frame, name, (x,y-10), font, 1, color, stroke, cv2.LINE_AA)
			
			img_item = 'images/'+ name + "/" + name + "_" +str(cont)+".png"
			cv2.imwrite(img_item, roi_gray)
			#cv2.imwrite(img_item, roi_color)
			color = (255,0,0)
			stroke = 2
			width = x + w
			heigh = y + h
			cv2.rectangle(frame, (x,y), (width, heigh), color, stroke)

		cv2.imshow('frame',frame)
		
		if cv2.waitKey(20) & 0xFF == ord('q'):
			break


while True:
	op = input("Gravar Pessoa? Y/N\n")

	if op.lower() == 'y':
		grava_face()
	elif op.lower() == 'n':
		break

treina(image_dir, face_cascade, recognizer)
reconhece()

cap.release()
cv2.destroyAllWindows()
