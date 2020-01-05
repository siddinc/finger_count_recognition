# Real-Time Finger Count Recognition
---
### Tech used: TensorFlow 2.0, OpenCV3, Python 3.5
### Trained Models:
---
`model2.h5` has the following accuracy metrics:
  - **training_accuracy = 99.62%**
  - **validation_accuracy = 100.00%**
> `model1.h5` is not used in the code as it has lower training and validation accuracy than `model2.h5`
### Instructions to run:
---
- Install [Anaconda](https://docs.anaconda.com/anaconda/install)
- `mkdir datasets` in the same directory as `src`
- Download the [Fingers Dataset](https://www.kaggle.com/koryakinp/fingers) into `datasets`
- Run `conda env create -f recog.yml`
- Run `conda activate recog`
- `cd` to `src`
- Run `python main.py`

> To obtain accurate results, orient your Webcam in such a way that your background is of a lighter hue; preferably white.