import cv2

capture = cv2.VideoCapture(0)
top, right, bottom, left = 70, 350, 285, 565
    
while(True):
    try:
        (check, frame) = capture.read()
        frame = cv2.flip(frame, 1)
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.imshow("Gray", gray)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") :
            capture.release()
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        capture.release()
        cv2.destroyAllWindows()
        break