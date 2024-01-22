import gradio as gr
from fastai.vision.all import *


def classify_image(img):
    learn = load_learner('model.pkl')
    img = PILImage.create(img)
    img.thumbnail((128, 128))
    pred, idx, probs = learn.predict(img)

    if pred == 'bear':
        result = f"The image is a Bear with probability {probs[idx]:.04f}"
    if pred == 'forest':
        result = f"The image is a Forest with probability {probs[idx]:.04f}"
    return result

example = ["bear.jpg", "forest.jpg"]
iface = gr.Interface(classify_image, inputs="image", outputs="text", examples=example)
iface.launch(share=True)
