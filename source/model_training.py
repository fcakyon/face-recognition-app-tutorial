# import the necessary packages
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

def create_mlp_model(optimizer='adam', neuron_number=50, lr=0.001, class_number=5):
    # Build function for keras/tensorflow based multi layer perceptron implementation
    model = Sequential()
    model.add(Input(shape=(128,)))
    model.add(Dense(neuron_number, activation='relu'))
    model.add(Dense(neuron_number, activation='relu'))
    model.add(Dense(class_number, activation='softmax'))
    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_mlp_model(embeddings_path = "", classifier_model_path = "", label_encoder_path = ""):
    # Trains a MLP classifier using embedding file from "embeddings_path", 
    # then saves the trained model as "classifier_model_path" and 
    # label encoding as "label_encoder_path".
    
    # Load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())
    
    # Encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    class_number = len(set(labels))
    
    # Reshape the data
    embedding_mtx = np.zeros([len(data["embeddings"]),len(data["embeddings"][0])])
    for ind in range(1,len(data["embeddings"])):
        embedding_mtx[ind,:] = data["embeddings"][ind]
        
    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    mlp_model = create_mlp_model(optimizer='adam', neuron_number=32, lr=1e-3, class_number=class_number)
    recognizer = KerasClassifier(model=mlp_model, 
                                 epochs=200, 
                                 batch_size=64, 
                                 verbose=1)
    
    recognizer.fit(embedding_mtx, 
                   labels)
    
    print("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)
    
    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)
        
def train_svm_model(embeddings_path = "", classifier_model_path = "", label_encoder_path = ""):
    # Trains a SVM classifier using embedding file from "embeddings_path", 
    # then saves the trained model as "classifier_model_path" and 
    # label encoding as "label_encoder_path".
    
    # Load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())
    
    # Encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    
    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    
    print("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)
    
    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)
        
def train_nb_model(embeddings_path = "", classifier_model_path = "", label_encoder_path = ""):
    # Trains a NB classifier using embedding file from "embeddings_path", 
    # then saves the trained model as "classifier_model_path" and 
    # label encoding as "label_encoder_path".
    
    # Load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())
    
    # Encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    
    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = GaussianNB()
    recognizer.fit(data["embeddings"], labels)
    
    print("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)
    
    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)
    