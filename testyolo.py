import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time

options = {
    'model': 'cfg/tiny-yolo-voc-5c.cfg',
    'load': 6875,
    'threshold': 0.0131
}
#
tfnet = TFNet(options)
#
# img = cv2.imread('bed4.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # use YOLO to predict the image
# result = tfnet.return_predict(img)
#
# img.shape
# tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
# br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
# label = result[0]['label']
#
#
# # add the box and label and display it
# img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
# img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
# plt.imshow(img)
# plt.show()

#
capture = cv2.VideoCapture('testvi'
                           'do.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while(capture .isOpened):
    start_time = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            t1 = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, t1, br, color, 7)
            frame = cv2.putText(frame, label, br, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.2f}'.format(1/(time.time() - start_time)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
