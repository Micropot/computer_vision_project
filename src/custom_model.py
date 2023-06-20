from transformers import AutoModelForImageClassification,  AutoImageProcessor
import torch
from PIL import Image
import Parameters

# dataset = load_dataset("mnist")
def scan_photo(file_path):
    image = file_path

    # open method used to open different extension image file
    im = Image.open(image)
    im = im.convert('RGB')

    model = AutoModelForImageClassification.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")
    image_processor = AutoImageProcessor.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")

    inputs = image_processor(im, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    # Parameters.prediction = model.config.id2label[predicted_label]
    pred = model.config.id2label[predicted_label]
    print(pred)
    return pred
