import cv2
import constants
import model_fn
import utils


loaded_model = model_fn.load_saved_model('fingers_model.h5')


def main():
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        try:
            (check, frame) = capture.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, constants.FRAME_DIM, interpolation=cv2.INTER_LINEAR)

            cv2.rectangle(frame, (constants.LEFT, constants.TOP),
                          (constants.RIGHT, constants.BOTTOM), (0, 255, 0), 2)
            cv2.imshow("Video Feed", frame)

            image = utils.preprocess_image(frame)
            print(image.shape)

            # (label, score) = model_fn.predict_image(loaded_model, image)
            # print('Predicted_label: {}  Score: {}'.format(label, score))

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
