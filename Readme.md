# Real-Time Finger Count Recognition
The goal of this project is to build and train a model which is able to count the number of fingers in real-time.

## Tech used:
- TensorFlow 2.0.0
- OpenCV 3.1.0
- Python 3.5.6

## Dataset:
- [Fingers Dataset](https://www.kaggle.com/koryakinp/fingers) used for training and testing
- 21600 images of fingers of left and right hands
- All images are grayscale with dimensions 128 x 128 pixels
- Training set: 18000 images
- Test set: 3600 images
- Number of classes: 6
> Images are centered by the center of mass with noise pattern in the background.

## Trained Models:
`model2.h5` has the following accuracy metrics:
  - Training accuracy = 99.62%
  - Validation accuracy = 100.00%
> `model2.h5` was trained for 20 epochs with a batch size of 180

## Instructions to run:
- Using `anaconda`:
  - Run `conda create --name <env_name> --file recog.yml`
  - Run `conda activate <env_name>`
- Using `pip`:
  - Run `pip install -r requirements.txt`
- `mkdir datasets` in the same directory as `src`
- Download the [Fingers Dataset](https://www.kaggle.com/koryakinp/fingers) into `datasets`
- `cd` to `src`
- Run `python main.py`

> To obtain accurate results, orient your Webcam in such a way that the background inside the green box (region of interest) is of a lighter hue; preferably white.