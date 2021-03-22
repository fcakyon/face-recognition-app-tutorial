# Flask Face Recognition

<a href="https://github.com/fcakyon/face-recognition-app-tutorial/actions/workflows/ci.yml"><img src="https://github.com/fcakyon/face-recognition-app-tutorial/actions/workflows/ci.yml/badge.svg" alt="CI Tests"></a>

A face recognition API using Python, Flask, Opencv, Pytorch, Heroku.

Live demo: https://face-recognition-api-flask.herokuapp.com (Temporarily unavailable due to [this issue](https://github.com/adriangb/scikeras/issues/221))

[Tutorial notebook](/tutorial/tutorial.ipynb) | [Tutorial presentation](/presentation/FaceRecognitionWebAppTutorial.pdf)

Refer to [tf2 branch](https://github.com/fcakyon/face-recognition-app-tutorial/tree/tf2) for tensorflow v2 support.

![DemoScreen](/images/webappscreen.jpg)

# App Usage

Run face detection app from [face-detection-app-tutorial repo](https://github.com/fcakyon/face-detection-app-tutorial):

```console
git clone https://github.com/fcakyon/face-detection-app-tutorial.git
cd face-detection-app-tutorial
python app.py
```

Then run face recognition app from this repo:

```console
git clone https://github.com/fcakyon/face-recognition-app-tutorial.git
cd face-recognition-app-tutorial
python app.py
```
