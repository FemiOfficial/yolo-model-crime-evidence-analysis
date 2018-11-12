import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

# InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/tiny-yolo.cfg',
    'load': 'bin/yolov2-tiny.weights',
    'threshold': 0.005
}

tfnet = TFNet(options)

img = cv2.imread('thNDEXDADG.jpg')
result = tfnet.return_predict(img)
print(result)

