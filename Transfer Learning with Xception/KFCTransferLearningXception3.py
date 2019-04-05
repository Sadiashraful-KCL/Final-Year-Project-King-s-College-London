from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras import optimizers
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
# from __future__import print_function, division
import keras


image_format = (299, 299)
trainingDirectory = 'Training'
validationDirectory = 'Testing'
image_files = glob(trainingDirectory + '/*/*.jp*g')
valid_image_files = glob(validationDirectory + '/*/*.jp*g')
imageNetWeights = 'imagenet'
reluAlgorithm = 'relu'
softmaxAlgorithm = 'softmax'
categoricalCrossentropyAlgorithm = 'categorical_crossentropy'
adamOptimizer = 'adam'
trainingBatchSize = 100
testingBatchSize = 50
epochNumber = 5
epochSteps = 10
classificationClasses = 8
colorMode = 'rgb'
validationAccuracy = 'val_acc'
validationLoss = 'val_loss'
trainingAccuracy = 'acc'
trainingLoss = 'loss'

preTrainedAlgorithm = Xception(input_shape = (299, 299, 3), weights = imageNetWeights,
                   include_top = False)

x = preTrainedAlgorithm.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

prediction = Dense(classificationClasses, activation = softmaxAlgorithmsoft)(x)
customModel = Model(inputs = preTrainedAlgorithm.input, outputs = prediction)
customModel.layers[0].trainable = False
customModel.summary()

callbacks_list = [
                  keras.callbacks.ModelCheckpoint(
                                                  filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                                                  monitor= validationLoss, save_best_only=True),
                  keras.callbacks.EarlyStopping(monitor= validationAccuracy, patience=1)
                  ]

customModel.compile(loss = categoricalCrossentropyAlgorithm,
              optimizer = adamOptimizer,
              metrics = ['accuracy'])


trainingDataGenerator = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,
                                           horizontal_flip = True, rotation_range = 20)

testingDataGenerator = ImageDataGenerator(rescale = 1./255)

# get label mapping for confusion matrix plot later
#test_gen = trainingDataGenerator.flow_from_directory(validationDirectory, target_size=(299, 299))
#labels = [None] * len(test_gen.class_indices)
#for k, v in test_gen.class_indices.items():
#  labels[v] = k
##
## should be a strangely colored image (due to VGG weights being BGR)
#for x, y in test_gen:
#  plt.title(labels[np.argmax(y[0])])
#  plt.imshow(x[0])
#  plt.show()
#  break

augmentedTrainingData = trainingDataGenerator.flow_from_directory('Training',
                                                 target_size = image_format,
                                                 batch_size = trainingBatchSize,
                                                 class_mode = 'categorical',
                                                 color_mode= colorMode,
                                                 shuffle=True)

augmentedTestingData = testingDataGenerator.flow_from_directory('Testing',
                                            target_size = image_format,
                                            batch_size = testingBatchSize,
                                            class_mode = 'categorical',
                                            color_mode = colorMode,
                                            shuffle = False)

#r = customModel.fit_generator(augmentedTrainingData, validation_data = augmentedTestingData,
#                       epochs = epochNumber, steps_per_epoch = epochSteps,
#                       validation_steps = len(augmentedTestingData))


#def get_confusion_matrix(data_path, N, trainingBatchSize):
#  # we need to see the data in the same order
#  # for both predictions and targets
#  print("Generating confusion matrix", N)
#  predictions = []
#  targets = []
#  i = 0
#  for x, y in trainingDataGenerator.flow_from_directory(data_path, 
#                                                        target_size=(299, 299), 
#                                                        shuffle=False, 
#                                                        batch_size=trainingBatchSize * 2):
#    
#    i += 1
#    if i % 50 == 0:
#      print(i)
#    p = customModel.predict(x)
#    p = np.argmax(p, axis=1)
#    y = np.argmax(y, axis=1)
#    predictions = np.concatenate((predictions, p))
#    targets = np.concatenate((targets, y))
#    if len(targets) >= N:
#      break
#  cm = confusion_matrix(targets, predictions)
#  return cm
#
#cm = get_confusion_matrix(trainingDirectory, len(image_files), trainingBatchSize)
#print(cm)
#valid_cm = get_confusion_matrix(validationDirectory, len(valid_image_files), testingBatchSize)
#print(valid_cm)

# accuracies
#plt.plot(r.history['acc'], label='train acc')
#plt.plot(r.history['val_acc'], label='val acc')
#plt.legend()
#plt.show()
#
## loss
#plt.plot(r.history['loss'], label='train loss')
#plt.plot(r.history['val_loss'], label='val loss')
#plt.legend()
#plt.show()
#
#import tensorflow as tf
#from keras.models import load_model
#customModel.save('KFC_new_model4.h5')
