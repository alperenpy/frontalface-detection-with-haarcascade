import cv2

cap=cv2.VideoCapture(1)
mycascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    prit=mycascade.detectMultiScale(gray,1.3,3)
    for (x,y,w,h) in prit:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,"face",(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)

    cv2.imshow("FrontalFaceDetector", frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
