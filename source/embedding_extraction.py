import os
import cv2
import pickle
from imutils import paths

def extract_embeddings(classes_dir = "", embeddings_path = ""):
    # Get models directory
    models_dir = "models" + os.sep
    
    # load our serialized face embedding model from disk
    print("[INFO] loading face embedding extractor...")
    face_embedding_model_filename = "openface_nn4.small2.v1.t7"
    embedder = cv2.dnn.readNetFromTorch(models_dir + face_embedding_model_filename)
    
    # Grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(classes_dir))

    # Initialize our lists of extracted facial embeddings and
    # corresponding character names
    known_embeddings = []
    known_names = []
    
    # Initialize the total number of faces processed
    total = 0
    
    # Loop over the image paths
    for (i, image_path) in enumerate(image_paths):
        
    	# Extract the person name from the image path
        name = image_path.split(os.sep)[-2]
        
        print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)) + " - " + image_path.split(os.sep)[-1])
        
    	# Load the image
        face = cv2.imread(image_path)

        # Construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
             (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        vec = embedder.forward()

		# Add the name of the person + corresponding face
		# embedding to their respective lists
        known_names.append(name)
        known_embeddings.append(vec.flatten())
        total += 1
    
    # Dump the facial embeddings + names to disk as pickle
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": known_embeddings, "names": known_names}
    with open(embeddings_path, "wb") as write_file:
        pickle.dump(data, write_file)