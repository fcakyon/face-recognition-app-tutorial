import os
import cv2
import imutils
from source.utils import draw_rectangles
from source.embedding_extraction import extract_embeddings
from source.face_recognition import recognize_faces, extract_faces
from source.model_training import train_mlp_model, train_svm_model, train_nb_model

# extract faces
raw_image_dir = "images" + os.sep + "lotr" + os.sep + "train" + os.sep + "raw"
extract_faces(raw_image_dir)

# extract embeddings
classes_dir = "images" + os.sep + "lotr" + os.sep + "train" + os.sep + "10_classes"
embeddings_path = "images" + os.sep + "lotr" + os.sep + "train" + os.sep + "embeddings.pickle"
extract_embeddings(classes_dir, embeddings_path)

# train nb classifier
classifier_model_path = "models" + os.sep + "lotr_nb_recognizer.pickle"
label_encoder_path = "models" + os.sep + "lotr_nb_le.pickle"
train_nb_model(embeddings_path, classifier_model_path, label_encoder_path)

# train svm classifier
classifier_model_path = "models" + os.sep + "lotr_svm_recognizer.pickle"
label_encoder_path = "models" + os.sep + "lotr_svm_le.pickle"
train_svm_model(embeddings_path, classifier_model_path, label_encoder_path)

# train mlp classifier
classifier_model_path = "models" + os.sep + "lotr_mlp_recognizer.pickle"
label_encoder_path = "models" + os.sep + "lotr_mlp_le.pickle"
train_mlp_model(embeddings_path, classifier_model_path, label_encoder_path)

# recognize face
file_path = "images" + os.sep + "lotr" + os.sep + "test" + os.sep + "raw" + os.sep +"legolas6.jpg"
image = cv2.imread(file_path)
image = imutils.resize(image, width=600)
recognitions = recognize_faces(image, classifier_model_path, label_encoder_path)

draw_rectangles(image, recognitions)
cv2.imshow("recognition result",image)
cv2.waitKey()
