import cv2,time
import os
os.environ['DISPLAY'] = ':0'  

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

a = 1 # holds number of frames

while True:
    a = a+1
    check,frame = video.read()
    print(frame)

    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)
    
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame , (x,y) , (x+w,y+h), (0,255,0), 3)

    cv2.imshow('capturing',frame)

    key = cv2.waitKey(1)      # for 1 millsec new frame is captured
    if key == ord('q'):       # exits when q is pressed
        break

print('Number of Frames captured : ',a)
video.release()
cv2.destroyAllWindows()
