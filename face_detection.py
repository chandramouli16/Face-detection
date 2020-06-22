import cv2
import os
os.environ['DISPLAY'] = ':0'                # for avoiding x server connection issue

# reading image

img  = cv2.imread('/home/cn7/courses/Computer Vision/sample1.jpeg',1)

'''print(type(img))

print("The size of img is : ",img.shape())

# resizing image

resized  = cv2.resize(img , (int(img.shape[1]/2)+200,int(img.shape[0]/2)+200)) 

print("The size of img is : ",resized.shape)

# to diaplay image

cv2.imshow('Sample',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# Cascade Classifier having face features
face_cascade = cv2.CascadeClassifier('/home/cn7/courses/Computer Vision/frontalFace10/haarcascade_frontalface_default.xml')
#ceating gray image of original image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# detecting cordinates of the face in total image window ,stores in numpy array
faces  = face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5)

print(type(faces))
print(faces)

#creating rectangle box for face 
for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    
#displaying image
cv2.imshow('Sample',img)
cv2.waitKey(0)
cv2.destroyAllWindows()