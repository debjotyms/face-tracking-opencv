import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    frame = cv2.copyMakeBorder(frame,250,0,20,20,cv2.BORDER_CONSTANT,value=(255,255,255,0))
    
    logo = cv2.imread('logo.png')
    logo = cv2.resize(logo, (150, 150))
    frame[:150, 270:420] = logo[:151,:151]
    frame = cv2.putText(frame, f'Face Detection and Tracking',(110,180),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in f:
        frame = cv2.putText(frame,f'Coordinates of Face ({x+(w//2)},{y+(h//2)})',(115,220),cv2.FONT_HERSHEY_DUPLEX,.9,(0,0,0),2)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        cv2.circle(frame,(x+(w//2),y+(h//2)),5,(255,255,255),-1)

        frame = cv2.putText(frame,f'Face',(x+w//2-25,y-10),cv2.FONT_HERSHEY_DUPLEX,.7,(255,255,255),1)
        
    cv2.imshow('Mongol-Tori Task', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
