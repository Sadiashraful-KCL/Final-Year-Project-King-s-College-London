import keras
from keras.layers import Dense
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.xception import Xception
#from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
#from keras.layers import Conv2D
# from __future__import print_function, division
from glob import glob

image_format = (299, 299)
inputShape = (299, 299, 3)
trainingDirectory = 'Training'
validationDirectory = 'Testing'
imageNetWeights = 'imagenet'
reluAlgorithm = 'relu'
softmaxAlgorithm = 'softmax'
categoricalCrossentropyAlgorithm = 'categorical_crossentropy'
adamOptimizer = 'adam'
trainingBatchSize = 128
testingBatchSize = 64
epochNumber = 10
epochSteps = 10
classificationClasses = 8
colorMode = 'rgb'
validationAccuracy = 'val_acc'
validationLoss = 'val_loss'
trainingAccuracy = 'acc'
trainingLoss = 'loss'


preTrainedAlgorithm = Xception(input_shape = inputShape, weights = imageNetWeights,
                   include_top = False)

for layer in preTrainedAlgorithm.layers[:-20]:
    layer.trainable = False
    
#for layer in preTrainedAlgorithm.layers:
#    layer.trainable = False

x = preTrainedAlgorithm.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation= reluAlgorithm)(x)
x = Dense(512, activation= reluAlgorithm)(x)

customModel.summary()

prediction = Dense(classificationClasses, activation = softmaxAlgorithm)(x)
customModel = Model(inputs = preTrainedAlgorithm.input, outputs = prediction)


callbacks_list = [
                  keras.callbacks.ModelCheckpoint(
                                                  filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                                                  monitor= validationLoss, save_best_only=True, mode= min, period= 0),
                  keras.callbacks.EarlyStopping(monitor= validationAccuracy, patience= 0, mode= max)
                  ]

customModel.compile(loss = categoricalCrossentropyAlgorithm,
              optimizer = adamOptimizer,
              metrics = ['accuracy'])


trainingDataGenerator = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True, rotation_range = 20)

testingDataGenerator = ImageDataGenerator(rescale = 1./255)

augmentedTrainingData = trainingDataGenerator.flow_from_directory(trainingDirectory,
                                                 target_size = image_format,
                                                 batch_size = trainingBatchSize,
                                                 class_mode = 'categorical',
                                                 color_mode= colorMode,
                                                 shuffle=True)

augmentedTestingData = testingDataGenerator.flow_from_directory(validationDirectory,
                                            target_size = image_format,
                                            batch_size = testingBatchSize,
                                            class_mode = 'categorical',
                                            color_mode = colorMode,
                                            shuffle = False)

r = customModel.fit_generator(augmentedTrainingData, validation_data = augmentedTestingData,
                      epochs = epochNumber, steps_per_epoch = epochSteps,
                      validation_steps = len(augmentedTestingData), callbacks= callbacks_list, verbose =1)

#for layer in customModel.layers[:-12]:
#    layer.trainable = False
    
#customModel.compile(loss = categoricalCrossentropyAlgorithm,
#              optimizer = adamOptimizer,
 #             metrics = ['accuracy'])

#r = customModel.fit_generator(augmentedTrainingData, validation_data = augmentedTestingData,
 #                     epochs = epochNumber, steps_per_epoch = epochSteps,
  #                    validation_steps = len(augmentedTestingData), callbacks= callbacks_list)

    
#accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

import tensorflow as tf
from keras.models import load_model
customModel.save('KFC_new_model12.h5')

Y_pred = customModel.predict_generator(augmentedTestingData, len(augmentedTestingData))
y_pred = np.argmax(Y_pred, axis = 1)
print('Confusion Matrix')
print(confusion_matrix(augmentedTestingData.classes, y_pred))
print('Classification Report')
classificationClasses = ['KFC Chicken(Thigh) 285 kcal',
                         'KFC Fillet Rice Box 448 kcal',
                         'KFC Fillet Tower Burger 650 kcal',
                         'KFC Fries 250 kcal',
                         'KFC Popcorn Chicken 285 kcal',
                         'KFC Zinger Burger 450 kcal',
                         'KFC Zinger Stacker Burger 780 kcal',
                         'Mcdonalds Big Mac Burger 508 kcal']
        
print(classification_report(augmentedTestingData.classes, y_pred, 
                            target_names = classificationClasses))


image_files = glob(trainingDirectory + '/*/*.jp*g')
valid_image_files = glob(validationDirectory + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(trainingDirectory + '/*')

# get label mapping for confusion matrix plot later
#test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
#print(test_gen.class_indices)
print(augmentedTestingData.class_indices)
labels = [None] * len(augmentedTestingData.class_indices)
for k, v in augmentedTestingData.class_indices.items():
  labels[v] = k
  
def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in testingDataGenerator.flow_from_directory(data_path, target_size=image_format, shuffle=False, batch_size=testingBatchSize * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = customModel.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(trainingDirectory, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(validationDirectory, len(valid_image_files))
print(valid_cm)

import util
from util import plot_confusion_matrix
plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')



































