import unittest
from io import BytesIO

from app import app

app.config["DETECTION_API_URL"] = "https://face-detection-api-flask.herokuapp.com/"

class TestApiEndpoints(unittest.TestCase):
    def test_recognize_with_3ch(self):
        client = app.test_client()

        # test image
        image_path = "tests/data/gollum4.jpg"
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        response = client.post(
            "/recognize", data = {'image': (BytesIO(image_bytes), 'gollum4.jpg')}, content_type='multipart/form-data',
        )

        self.assertEqual(len(response.json["recognitions"]), 3)
