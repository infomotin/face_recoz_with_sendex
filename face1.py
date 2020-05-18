import face_recognition_models as face_recognition
# import face
import os
import cv2
try:
	print("face_recognition version:")
	print(face_recognition.__version__)
except Exception as e:
	print(e)


KNOWN_FACES_DIR = "know_faces"
UNKNOWN_FACES_DIR = "unknow_faces"
TOLEREANCE = 0.6
FACE_THICKNESS = 3
FONT_THINKENESS = 2
MODEL = "cnn"

print("Loading Knowa faces ---------")

knows_faces = []
knows_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        knows_faces.append(encoding)
        knows_names.append(name)
        
print("Prosseing Unknow Face -----")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image ,model=MODEL)
    encodings = face_recognition.face_encodings(image,locations)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    for face_encoding, face_location in zip(encodings,locations):
        results = face_recognition.compare_face(knows_faces,face_encoding,locations,TOLEREANCE)
        match = None
        if True in results:
            match = knows_names[results.index(True)] 
            print(f"Match Face :{match}")
            
            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1],face_location[2])
            
            color = [0,255,0]
            cv2.rectangle(image,top_left,bottom_right,color,FACE_THICKNESS)
            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1],face_location[2]+22)
            cv2.rectangle(image,top_left,bottom_right,color,FACE_THICKNESS,cv2.FILLED)
            cv2.putText(image,match,(face_location[3]+10,face_location[0]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,FONT_THINKENESS)
    cv2.imshow(filename,image)
    cv2.waitKey(10000)
    # cv2.destroyWindow(filename)