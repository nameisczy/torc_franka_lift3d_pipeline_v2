import os
import json
import base64

import cv2
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM-Instruct")
messages = [
    {
        "role":
        "user",
        "content": [
            {
                "type":
                "image",
                "url":
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
            }, {
                "type": "text",
                "text": "What animal is on the candy?"
            }
        ]
    },
]
out = pipe(text=messages)[0]['generated_text'][-1]
print(json.dumps(out, indent=2))

curdir = os.path.dirname(os.path.abspath(__file__))
img_from_file = cv2.imread(curdir + '/candy.jpg')
retval, buffer = cv2.imencode('.jpg', img_from_file)
img_b64 = base64.b64encode(buffer).decode('utf-8')
# print(img_b64)

messages = [
    {
        "role":
        "user",
        "content": [
            {
                "type": "image",
                "base64": img_b64
            }, {
                "type": "text",
                "text": "What animal is on the candy?"
            }
        ]
    },
]
out = pipe(text=messages)[0]['generated_text'][-1]
print(json.dumps(out, indent=2))
