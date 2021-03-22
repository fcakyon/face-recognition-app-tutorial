import os
from flask import Flask,jsonify,request,render_template
from source.face_recognition import recognize_faces
from source.utils import draw_rectangles, read_image, prepare_image
from source.model_training import create_mlp_model

app = Flask(__name__)

app.config.from_object('config')
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def detect():
    file = request.files['image']

    # Read image
    image = read_image(file)
    
    # Recognize faces
    classifier_model_path = "models" + os.sep + "lotr_mlp_10c_recognizer.pickle"
    label_encoder_path = "models" + os.sep + "lotr_mlp_10c_labelencoder.pickle"
    faces = recognize_faces(image, classifier_model_path, label_encoder_path, detection_api_url=app.config["DETECTION_API_URL"])

    return jsonify(recognitions = faces)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    # Read image
    image = read_image(file)
    
    # Recognize faces
    classifier_model_path = "models" + os.sep + "lotr_mlp_10c_recognizer.pickle"
    label_encoder_path = "models" + os.sep + "lotr_mlp_10c_labelencoder.pickle"
    faces = recognize_faces(image, classifier_model_path, label_encoder_path, detection_api_url=app.config["DETECTION_API_URL"])
    
    # Draw detection rects
    draw_rectangles(image, faces)
    
    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('index.html', face_recognized=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)

if __name__ == '__main__':
    app.run(debug=True,
            use_reloader=True,
            port=4000)
