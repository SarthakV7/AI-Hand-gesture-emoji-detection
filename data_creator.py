import cv2
import numpy as np

def contours(diff ,th=105):
    _ , thresholded = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
    return (thresholded, hand_segment)

#****************************************************************************************************
tk = [cv2.TrackerBoosting_create(),cv2.TrackerMIL_create(), cv2.TrackerKCF_create(), cv2.TrackerTLD_create(), cv2.TrackerCSRT_create()]
tracker = tk[2]
#1.good, 2.best, 3.poor(resizing), 4.very accurate

#****************************************************************************************************
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
#     cv2.putText(frame, 'press space when ready!', (20,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (25,155,255))
    cv2.rectangle(frame, (350,100), (500,250), (0,255,0), 3)
    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.destroyAllWindows()
        break
# roi = cv2.selectROI(frame, False)
roi = (350, 100, 150, 150)
print(' ROI -------> ', roi)
ret = tracker.init(frame, roi)
cv2.destroyAllWindows()

c = 0
d = c+1

while True:
#     try:
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
        hand = contours(gray, th=150)[0]
#             result = pred(hand) ####
        cv2.rectangle(frame, pt1, pt2, (255,255,0), 3)

        cv2.imshow('thresholded', hand)
        hand_flip = cv2.flip(hand, 1)
        cv2.imwrite(f'./mydata/seven/{c}.jpg', hand)
        cv2.imwrite(f'./mydata/seven/flip/{d}.jpg', hand_flip)
        c+=2
        d+=2
        if c%500 == 0:
            print('****************************\n***************************\n****************************\n****************************\n****************************\n***************************\n')
        print('value of c,d = ', (c,d))
    else:
        cv2.putText(frame, 'Failed to detect object', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        
    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
