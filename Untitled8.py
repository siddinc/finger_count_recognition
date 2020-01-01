from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import one_hot
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import os

epochs = 20
batch_size = 180
train_path = os.path.abspath('./machine_learning_examples/datasets/fingers/train')
test_path = os.path.abspath('./machine_learning_examples/datasets/fingers/test')

from glob import glob
train_dataset = glob(train_path + '/*.png')
test_dataset = glob(test_path + '/*png')

def load_training_data(train_dataset):
    x_train = []
    y_train = []
    
    for img in train_dataset:
        y_train.append(int(img[-6]))
        np_img = np.array(Image.open(img), dtype='uint8')
        np_img = np.reshape(np_img, (128, 128, -1))
        x_train.append(np_img)
    x_train, y_train = np.array(x_train), np.array(y_train)
    return (x_train, y_train)


# In[7]:


def load_testing_data(test_dataset):
    x_test = []
    y_test = []
    
    for img in test_dataset:
        y_test.append(int(img[-6]))
        np_img = np.array(Image.open(img), dtype='uint8')
        np_img = np.reshape(np_img, (128, 128, -1))
        x_test.append(np_img)
    x_test, y_test = np.array(x_test), np.array(y_test)
    return (x_test, y_test)


# In[8]:


def preprocess_data(x_train, x_test, y_train, y_test):
    classes = len(set(y_train))
    x_train, x_test = x_train/x_train.max(), x_test/x_test.max()
    y_train, y_test = one_hot(y_train, classes), one_hot(y_test, classes)
    return x_train, x_test, y_train, y_test, classes


# In[ ]:


x_train, y_train = load_training_data(train_dataset)
x_test, y_test = load_testing_data(test_dataset)
x_train, x_test, y_train, y_test, classes = preprocess_data(x_train, x_test, y_train, y_test)


# In[ ]:


steps_per_epoch = x_train.shape[0] // batch_size


# In[9]:


i = Input(shape=x_train[0].shape)

x = Conv2D(64, (3,3), strides=(1,1), activation='relu')(i)
x = BatchNormalization()(x)

x = Conv2D(64, (3,3), strides=(1,1), activation='relu')(i)
x = BatchNormalization()(x)

x = MaxPool2D(pool_size=(4,4))(x)

x = Conv2D(128, (3,3), strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)

x = Conv2D(128, (3,3), strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)

x = MaxPool2D(pool_size=(8,8))(x)

x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(classes, activation='softmax')(x)

model = Model(i, x)


# In[10]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


data_generator = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
)


# In[ ]:


train_generator = data_generator.flow(x_train, y_train, batch_size=batch_size)


# In[58]:


r = model.fit_generator(train_generator, validation_data=(x_test, y_test), epochs=epochs)


# In[60]:


print(model.evaluate(x_test, y_test))


# In[61]:


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()


# In[62]:


plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()


# In[ ]:


p_test = model.predict(x_test).argmax(axis=1)
p_test = one_hot(p_test, classes)


# In[17]:


missclassified_ex_ind = np.where(p_test != y_test)[0]
print(missclassified_ex_ind)

def missclassification(missclassified_ex_ind):
  if len(missclassified_ex_ind) == 0:
    print("no missclassified labels")
    return
  for i in missclassified_ex_ind:
    plt.imshow(x_test[i].reshape(128,128))
    plt.title("True label: {} Predicted label: {}".format(y_test[i], p_test[i]));


# In[65]:


missclassification(missclassified_ex_ind)


# In[ ]:


get_ipython().system('pip install -q pyyaml h5py')


# In[ ]:





# In[2]:


loaded_model = tensorflow.keras.models.load_model('./machine_learning_examples/models/fingers_model.h5')


# In[14]:


loss, acc = loaded_model.evaluate(x_test, y_test)
print(loss, acc)


# In[ ]:


p_test = model.predict(x_test).argmax(axis=1)
p_test = one_hot(p_test, classes)
missclassified_ex_ind = np.where(p_test != y_test)[0]
print(missclassified_ex_ind)


# In[ ]:


missclassification(missclassified_ex_ind)


# In[ ]:




