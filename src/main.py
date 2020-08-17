import cv2
import constants
import model_fn
import utility_fn

# load pretrained model
loaded_model = model_fn.load_saved_model('model2.h5')


def main():
    # driver function
    # get webcam feed
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        try:
            (check, frame) = capture.read()
            # flip frame horizontally to get mirror image
            frame = cv2.flip(frame, 1)
            # resize frame to 640 X 480
            frame = cv2.resize(frame, constants.FRAME_DIM,
                               interpolation=cv2.INTER_LINEAR)
            # preprocess frame to extract ROI
            image = utility_fn.preprocess_image(frame)
            # obtain prediction from model
            (label, score) = model_fn.predict_image(loaded_model, image)
            # display roi and metrics on cloned frame
            utility_fn.display_frame(frame, label, score)

            # wait for user to press 'q'
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
