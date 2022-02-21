#Importation des bibliothèque utiles
import face_recognition
import pickle
import cv2
import imutils
import numpy as np
from datetime import datetime
# Chargement de notre base de données encodées
known_faces, known_names = pickle.loads(open('./dataset_encodings.pickle', "rb").read())
#face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# Affichons les noms présent dans notre base de données
#print(known_names)

# Initialisatqion des paramètre de ronnaissance de visage
MODEL = 'hog'
TOLERANCE = 0.6
#frame = imutils.resize(frame, width=640)
'''
# Création d'une Fonction d'enregistrement dans un un fichier csv
def enregData (name):
    with open('./enregistrement.csv', 'r+') as f:
        # Lecture de notre fichier d'enregistrement
        myData = f.readlines()
        nameList = []
        # Initialisons la liste des personnes à enrégistrer
        for line in myData:
            entree = line.split(',')
            nameList.append(entree[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dayString = now.strftime('%d-%m-%Y')
            f.writelines(f'\n{name},{dtString},{dayString}')
'''
# Ouverture de notre webCam
video_capture = cv2.VideoCapture('rtsp://192.168.1.110:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream')

############################################################
# ETAPE: RECONNAISSANCE DES VISAGES AVEC WEBAM
############################################################

while True:
    ret, frame = video_capture.read()
    frame = imutils.resize(frame, width=600)
    # Convertir l'image de la frame(fenetre) en RGB et gray scale
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgb_frame = frame[:, :, ::-1]

    # Localisation du visage et encodage des visage sur la webCam
    face_location_CurFrame = face_recognition.face_locations(rgb_frame, model=MODEL)
    #face_location_CurFrame = face_recognition.face_landmarks(rgb_frame)
    # Affichage du carré sur l'image détecté
    for face_location in face_location_CurFrame:
        # Affichage du carré sur le visage
        top, right, bottom, left = face_location
        print('top = ', top,'right :', right,'left : ', left,'bottom :', bottom)
        cv2.rectangle(frame,(left,top), (right,bottom),(0,0,255),2)

        # Encodage de l'image detecté
        face_encode = face_recognition.face_encodings(rgb_frame, [face_location])[0]

        # Comparaison entre les images visages détectés et ceux contenus dans ma bases d'images
        results = face_recognition.compare_faces(known_faces, face_encode, tolerance=TOLERANCE)
        faceDist = face_recognition.face_distance(known_faces, face_encode)

        # Affichage du nom de la personne si le visage est reconnu ou inconnu si le visage est inconnu
        # Récupération de l'index du visage qui a la la plus petite distance
        best_match_index = np.argmin(faceDist)
        if results[best_match_index]:
            name = known_names[best_match_index].upper()
            color = (0, 255, 255)
            cv2.rectangle(frame, (left, top - 35), (right, bottom), color, 2)
            cv2.putText(frame, name, (left + 6, bottom + 18), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, 2)
            # Enrégistrement des personne reconnonus dans un fichier excel.
            #date = enregData(name)

        else:
            name = 'unknown'
            #color = (0, 0, 255)
            #cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom + 18), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        #cv2.putText(frame, name, (left + 6, bottom + 18), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 255), 2)

    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()