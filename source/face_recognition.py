import os
import cv2
import pickle
import requests
import numpy as np
from imutils import paths

def detect_faces(image, detection_api_url="http://127.0.0.1:3000/"):
    # Apply previously implemented deep learning-based face detector to 
    # localize faces in the input image
    image = cv2.imencode('.jpg', image)[1].tostring()

    # Send request to detection_api_url
    if not detection_api_url[-1]=="/":
        detection_api_url += "/"
    try:
        response = requests.post(detection_api_url+"detect", files={'image': image})
    except:
        Exception("You need to run detection app as described in https://github.com/fcakyon/face-recognition-app-tutorial#app-usage")
    
    # Convert response to json object (dictionary)
    response_json = response.json()
    # Convert response to json object (dictionary)
    detections = response_json["detections"]
    return detections

def extract_faces(raw_image_dir = "", detection_api_url="http://127.0.0.1:3000/"):
    # Extract faces from the images in images/base folder
    min_confidence = 30
    
    # Create export dir
    faces_dir = raw_image_dir + os.sep + ".." + os.sep + 'faces'
    os.makedirs(faces_dir, exist_ok=True)
    
    # Grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(raw_image_dir))
    
    index = 0
    # Loop over the image paths
    for (i, image_path) in enumerate(image_paths):
    	# Extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
    		len(image_paths)))
    
    	# Load the image and then resize it
        image = cv2.imread(image_path)
        # Get image shape
        (image_height, image_width) = image.shape[:2]
        resized_image = cv2.resize(image, (300, 300))
        #image = resize(image, width=600)
    
        print("[INFO] performing face detection over api for: " + image_path.split("\\")[-1])
        detections = detect_faces(resized_image, detection_api_url)
        #detections = detect_faces(image)
        
        # Ensure at least one face was found
        if len(detections) > 0:
            for detection in detections:
                # Extract the confidence (i.e., probability) associated with the
                # prediction 
                confidence = detection["prob"]
                
                # Get detection coords
                [start_x, start_y, end_x, end_y] = detection["rect"]
                # Correct the detections regions
                start_x = int(start_x/300*image_width)
                start_y = int(start_y/300*image_height)
                end_x = int(end_x/300*image_width)
                end_y = int(end_y/300*image_height)
                
                # Ensure that the detection with the largest probability also
            	  # means our minimum probability test (thus helping filter out
            	  # weak detections)
                if confidence > min_confidence:
                    # Extract the face ROI
                    face = image[start_y :end_y, start_x : end_x]
                    (fH, fW) = face.shape[:2]
                    
                    # Ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue
                    
                    face_path = raw_image_dir + os.sep + ".." + os.sep + 'faces' + os.sep + "face" + '_' + str(index) + ".jpg"
                    print("face_path: " + face_path)
                    cv2.imwrite(face_path, face)
                    index = index + 1
    
def recognize_faces(image, classifier_model_path, label_encoder_path, detection_api_url="http://127.0.0.1:3000/"):
    '''Recognize faces in an image'''
    faces_list = []
    min_detection_confidence = 20 # percent
    
    # Load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    models_dir = "models" + os.sep
    face_embedding_model_filename = "openface_nn4.small2.v1.t7"
    embedder = cv2.dnn.readNetFromTorch(models_dir + face_embedding_model_filename)
    
    # Load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(classifier_model_path, "rb").read())
    label_encoder = pickle.loads(open(label_encoder_path, "rb").read())
    
    print("[INFO] performing face detection over api...")
    detections = detect_faces(image, detection_api_url)
    
    print("[INFO] performing face recognition...")
    # Loop over the detections
    for detection in detections:
        # Get detection region
        [start_x, start_y, end_x, end_y] = detection["rect"]
        # Extract the confidence (i.e., probability) associated with the
        # prediction 
        detection_confidence = detection["prob"]
        
        # Filter out weak detections
        if detection_confidence > min_detection_confidence:
            # Extract the face ROI
            face = image[start_y:end_y, start_x:end_x]
            (fH, fW) = face.shape[:2]
            
            # Ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            
            # Construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                 (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(face_blob)
            vec = embedder.forward()
    
            # Perform classification to recognize the face
            preds = recognizer.predict_proba(vec)
            j = np.argmax(preds)
            # Get recognition confidence
            try:
                recognition_confidence = preds[j]
            except:
                recognition_confidence = preds[0][j]
            # Convert it to a native python variable (float)
            recognition_confidence = recognition_confidence.item()
            # Get recognition class name
            name = label_encoder.classes_[j]
            # Convert it to a native python variable (str)
            name = name.item()
    
            # Append results to list
            face_dict = {}
            face_dict['rect'] = [start_x, start_y, end_x, end_y]
            face_dict['detection_prob'] = detection_confidence
            face_dict['recognition_prob'] = recognition_confidence * 100
            face_dict['name'] = name
            faces_list.append(face_dict)
            
    # Return the face image area, the face rectangle, and face name
    return faces_list
