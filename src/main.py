import cv2
from .model import predict_image
from .utils import preprocess_image


def main():
  capture = cv2.VideoCapture(0)
  top, right, bottom, left = 70, 350, 285, 565
    
  while capture.isOpened():
    try:
      (check, frame) = capture.read()
      frame = cv2.flip(frame, 1)
      (height, width) = frame.shape[:2]

      roi = frame[top:bottom, right:left]
      gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      gray = cv2.resize(gray, (128, 128))

			np_gray = preprocess_image(gray)
			(label, score) = predict_image(np_gray)

      cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
      cv2.imshow("Frame", frame)
      if (cv2.waitKey(1) & 0xFF) == ord("q"):
        capture.release()
        cv2.destroyAllWindows()
        break

    except(KeyboardInterrupt):
      capture.release()
      cv2.destroyAllWindows()
      break

if __name__ == '__main__':
	main()