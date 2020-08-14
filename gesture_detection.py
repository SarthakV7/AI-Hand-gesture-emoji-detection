import cv2
import time
import numpy as np
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras import optimizers
from yolo import YOLO

# Metrics for checking the model performance while training
import tensorflow as tf

def create_model():
    K.clear_session()
    ip = Input(shape = (150,150,1))
    z = Conv2D(filters = 32, kernel_size = (64,64), padding='same', input_shape = (150,150,1), activation='relu')(ip)
    z = Conv2D(filters = 64, kernel_size = (16,16), padding='same', input_shape = (150,150,1), activation='relu')(z)
    z = Conv2D(filters = 128, kernel_size = (8,8), padding='same', input_shape = (150,150,1), activation='relu')(z)
    z = MaxPool2D(pool_size = (4,4))(z)
    z = Flatten()(z)
    z = Dense(32, activation='relu')(z)
    op = Dense(5, activation='softmax')(z)
    model = Model(inputs=ip, outputs=op)
    return model

model = create_model()
model.load_weights('model_weights.h5')

#****************************************************************************************************
def paste(frame, s_img, pt1, pt2):
    x, y = pt1
    x_w, y_h = pt2
    n = min(x_w-x, y_h-y)
    s_img = cv2.resize(s_img, dsize=(n,n))
    l_img = frame
    (x_offset,y_offset,_) = (frame.shape)
    (y_offset,x_offset) = (x_offset//2 - 72, y_offset//2 - 72)
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y:y_h, x:x_w, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y:y_h, x:x_w, c])
    return l_img

#****************************************************************************************************
def blur(frame, roi):
    (x,y,w,h) = tuple(map(int, roi))
    roi_ = frame[y:y+h, x:x+w, :]
    b = 20
    blurred_roi = cv2.blur(roi_, ksize=(b,b))
    frame[y:y+h, x:x+w, :] = blurred_roi
    return frame

#****************************************************************************************************
emoji = []
for i in range(5):
    img = cv2.imread(f'./emoji_data/emoji/{i+1}.png', -1)
    emoji.append(img)

#****************************************************************************************************
def icon(x, e = emoji):
    return e[x]

#****************************************************************************************************
def pred(img):
    return model.predict(img).argmax(axis=1)[0]

#****************************************************************************************************
def contours(diff ,th=100):
    _ , thresholded = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
    return (thresholded, hand_segment)

#****************************************************************************************************
tk = [cv2.TrackerBoosting_create(), cv2.TrackerMIL_create(), cv2.TrackerKCF_create(), cv2.TrackerTLD_create(), cv2.TrackerCSRT_create()]
tracker = tk[2]
#1.good, 2.best, 3.poor(resizing), 4.very accurate

#****************************************************************************************************

def main():
    roi = []
    while(len(roi) == 0):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = hand_detect(frame)

    cv2.destroyAllWindows()
    ret = tracker.init(frame, roi)
    cv2.destroyAllWindows()

    c = 0
    d = c+1

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        success, roi = tracker.update(frame)
        (x,y,w,h) = tuple(map(int, roi))
        if success:
            pt1 = (x,y)
            pt2 = (x+w, y+h)
            square  = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            gray_ = cv2.medianBlur(gray, 7)
            hand = contours(gray, th=100)[0]
            im = np.array([hand])
            im = im.reshape(-1,150,150,1)
            result = pred(im) ####
            if result == 7:
                result = 1

            cv2.rectangle(frame, pt1, pt2, (255,255,0), 3)
            emo = icon(result)

            frame_copy = paste(frame, emo, pt1, pt2) ####
            (a,b) = hand.shape
            for i in range(3):
                hand = hand*255
                frame_copy[0:a, 0:b, i] = hand

        else:
            cv2.putText(frame, 'Failed to detect object', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow('output', frame_copy)
            roi = []

            while(len(roi) == 0):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                roi = hand_detect(frame)
            continue
        cv2.imshow('output', frame_copy)
        print(frame.shape)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

cascade = cv2.CascadeClassifier('haarcascade/fist.xml')
cap = cv2.VideoCapture('video.mp4')

yolo = YOLO("./models/cross-hands.cfg", "./models/cross-hands.weights", ["hand"])
yolo.size = 416
yolo.confidence = 0.2

def hand_detect(img, type='haar'):

    cv2.imshow('output', img)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        cv2.destroyAllWindows()

    hand_img = img.copy()
    if type=='yolo':
        width, height, inference_time, results = yolo.inference(img)
    elif type=='haar':
        results = cascade.detectMultiScale(hand_img, scaleFactor = 1.3, minNeighbors = 2)


    if len(results) > 0:
        if type=='yolo':
            _,_,_, x, y, w, h = results[0]
            return x,y,150,150
        elif type=='haar':
            x, y, w, h = results[0]
            return x-50,y-70,150,150
    else:
        return []

main()
