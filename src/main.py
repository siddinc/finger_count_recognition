import cv2
import constants
import model_fn
import utility_fn


loaded_model = model_fn.load_saved_model('model2.h5')

def main():
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        try:
            (check, frame) = capture.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, constants.FRAME_DIM, interpolation=cv2.INTER_LINEAR)
            image = utility_fn.preprocess_image(frame)
            (label, score) = model_fn.predict_image(loaded_model, image)
            utility_fn.display_frame(frame, label, score)

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
