import tensorflow as tf
import tensorflow_hub as hub
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

path = "C:\\Users\\USER\\Desktop\\code\\vs11978241.jpg"
with open(path, 'rb') as f:
    data = f.read()
encoded_img = np.frombuffer(data, dtype = np.uint8)
frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
frame = cv2.resize(frame, (256, 192))

img = frame.copy()
img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
input_img = tf.cast(img, dtype=tf.int32)

results = movenet(input_img)
keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
print(keypoints_with_scores)

loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)

cv2.imshow('Movenet MuiltiPose', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()