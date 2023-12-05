import cv2


model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("output.avi")

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags= cv2.CASCADE_SCALE_IMAGE

    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255,255,0),2 ) ## rawing the rectangle aganist the face
    
    cv2.imshow("Faces",frame) ## Displaying detactable face image to user 

    
    if cv2.waitKey(1) == ord ("q"):
        break

camera.release()
cv2.destroyAllWindows()