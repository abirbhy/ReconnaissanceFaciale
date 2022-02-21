######################################################
###  ETAPE 1:IMPORTATION DES BIBLIOTHEQUES UTILES  ###
######################################################
import cv2
import pickle
import os
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

######################################################
### ETAPE 2: CHARGEMENT DES DONNEES SUR PYCHARM   ####
######################################################µ

# Dataset_path
#dataset_path = './Personnel_SFM'
dataset_path = './Dataset_sfm'
# Haarcascade_file
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
# CHARGEMET DES DONNEES

# Fonction de detection de visage
def detecte_face (img):
    # Convertir l'image en gris
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (600, 480))
    return img

# Initialisation de la liste des visages et des noms de chaque employés de SFM
# Liste des visages connus
known_faces = []
# Liste des noms connus
known_names = []

# Recherchons la liste des visages et des noms de chaques personne.
for name in os.listdir(dataset_path):
    # Téléchargement de tous les fichiers images de SFM
    for filename in os.listdir(f'{dataset_path}/{name}'):
        print(f'Processing images of: {name}')
        # Télécharger une images
        img = face_recognition.load_image_file(f'{dataset_path}/{name}/{filename}', mode='RGB')
        fm = cv2.Laplacian(img, cv2.CV_64F).var()
        if fm < 50:
            print("Photo exclue :",filename,fm)
        else:
            try:
                print("photo acceptée :",name,filename,fm)
                face_encoding = face_recognition.face_encodings(img)[0]
            except IndexError:
                continue
            # Ajoutons la liste des visages encodés ainsi que les noms
            known_faces.append(face_encoding)
            # Ajoutons les noms de notre dataset à la liste des nom des personnes de SFM
            known_names.append(name)

# Affichons la taille de chaque images
print(img.shape)
# Afficons le nombre de noms de notre liste de noms connus
print("Nombre de nom :", len(known_names))

# Affichons le nombre de visages de notre liste de visage
print("nombre de visage :", len(known_faces))

# Affichons la liste unique des personne de SFM
names = np.unique(known_names)
print(names)

# Enregistrons nos données dans un fichier pickle
data = [known_faces, known_names]
f = open('./dataset_encodings.pickle', "wb")
f.write(pickle.dumps(data))
f.close()


