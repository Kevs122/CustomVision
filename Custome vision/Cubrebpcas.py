from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

prediction_key = "a8242ea53f7148b884029cd8f516cc14"
ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/"
project_id = "d1c48a03-d759-45d5-a436-0c4905752990"
publish_iteration_name = "Cubrebocas"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

with open("C:/Users/Kevin/Desktop/Custome vision/Ima cubrebocas/descarga.jpg", mode="rb") as test_data:
    results = predictor.detect_image(project_id, publish_iteration_name, test_data)

for prediction in results.predictions:
    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))

