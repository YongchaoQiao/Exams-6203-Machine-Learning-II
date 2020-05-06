# # %%---------------------------------------Introduction---------------------------------------------------------------
# # My best model was found on day6. Actually, the best model is a ensemble model which combines the best model in day3
# and a high-score model in day6. All things I did to get my best model in Exam1 contains the work on day1, day2, day3
# and day6.
# Specifically, there are four steps:
# 1. On day1, I uploaded the train.zip to the cloud, read the images and targets, saved it in to the train folder in
# the cloud.
# 2. On day2, I calculated the proportion of each categories. Based on the proportion, then do the augmentation on the
# last three categories to get a general balanced total data. The operation here includes: rotation(179), horizontal
# flip, vertical flip, shear, feature-wise center and feature-wise_std_normalization.
# 3. On day3, I used the generated dataset named as x_total_image32three and target_total_image32three. This dataset is
# a augmented dataset, using all the information of the original dataset. Then I split it into train and test dataset to
# train the model. I built the MLP network and tuned the LR, N_NEURONS, N_EPOCHS, BATCH_SIZE and DROPOUT. Finally, I
# tested the model on the test set of day1 to see the performance and get the model with the highest score.
# 4. On day6, I used the generated dataset named as x_total_image32_six_no_test and target_total_image32_six_no_test
# as total set. This total set is a balanced augmented dataset, using all the information of the original whole dataset.
# Then I split into train and test set and built the MLP network, specified the class weight parameter tune the LR,
# N_NEURONS, N_EPOCHS, BATCH_SIZE and DROPOUT. Next, I tested the model on the test set of day1 to see the
# performance and get the models with the higher scores. Finally, I combined one high-score model with the best
# model in day3 to get the best ensemble model which is also my best model in Exam1.


# # %%---------------------------------------Import packages------------------------------------------------------------
import os
import re
import cv2
import random
import zipfile
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, Activation

# # %%---------------------------------------Data Preparation-----------------------------------------------------------
# Work of day1:
# Get the contents from the train.zip file
z = zipfile.ZipFile(r'/home/ubuntu/train.zip', 'r')

for f in z.namelist():
    item = z.open(f, 'r')
    z.extract(f, r'/home/ubuntu/Exam1')
z.close()

train_cell_dir = "/home/ubuntu/Exam1/train"


def read_directory(directory_name):
    image_id = []
    content_id = []
    for filename in sorted(os.listdir(directory_name)):
        if re.findall(r"\.png", filename):
            img = cv2.imread(directory_name + "/" + filename)
            image_id += [(img, filename.strip(r"\.png"))]
        elif not re.findall(r"\.png", filename):
            txtf = open(directory_name + "/" + filename)
            content = txtf.read()
            content_id += [(content, filename.strip(r"\.txt"))]

    return image_id, content_id


image_tuple, target_tuple = read_directory(train_cell_dir)

# Check whether the image is matched with the target
judgement = 0
for i in range(len(image_tuple)):
    if image_tuple[i][1] != target_tuple[i][1]:
        print('ERROR')
    else:
        judgement += 1
print(judgement)

# Resize the image array to (32, 32, 3)
res_day1 = np.zeros([len(image_tuple), 32, 32, 3])

for i in range(len(image_tuple)):
    res_day1[i] = cv2.resize(image_tuple[i][0], (32, 32), interpolation=cv2.INTER_CUBIC)

# Get the x_total with shape as (n_image, 3072)
x_total = res_day1.reshape(len(res_day1), -1)

# Get the target_total with shape as (n_image, 1)
target_total = np.zeros([len(target_tuple), 1])
for i in range(len(target_tuple)):
    if 'red blood cell' == target_tuple[i][0]:
        target_total[i] = 0
    elif 'ring' == target_tuple[i][0]:
        target_total[i] = 1
    elif 'schizont' == target_tuple[i][0]:
        target_total[i] = 2
    elif 'trophozoite' == target_tuple[i][0]:
        target_total[i] = 3

# Save the x_total and target_total as np arrays
# np.save("/home/ubuntu/Exam1/x_total.npy", x_total)
# np.save("/home/ubuntu/Exam1/target_total.npy", target_total)

# Work of day2
# Based on the by-product of day1, the proportion of four categories are: red blood cell: ring: schizont: trophozoite  =
# 7000: 365: 133: 1109 = 1: 19: 53 :6. Based on this proportion we can know the number of augmentation for each category
# Augment the total dataset for day3 and day6
# For day3, there are only three operations for the ImageDataGenerator: rotation, horizontal and vertical flip.
# For day6, there are six operations for the ImageDataGenerator: rotation, horizontal and vertical flip, shear,
# feature-wise_center and feature-wise_std_normalization.
# Data of Day3:
new_total_list_day3 = []
for j in range(len(image_tuple)):
    if target_tuple[j][0] == 'red blood cell':
        new_total_list_day3 += [(image_tuple[j][0], target_tuple[j][0])]
    elif target_tuple[j][0] == 'ring':
        new_total_list_day3 += [(image_tuple[j][0], target_tuple[j][0])]
        for i in range(18):
            image_generator = ImageDataGenerator(rotation_range=179, horizontal_flip=True, vertical_flip=True)
            new_total_list_day3 += [(image_generator.random_transform(image_tuple[j][0], seed=None), target_tuple[j][0])]
    elif target_tuple[j][0] == 'schizont':
        new_total_list_day3 += [(image_tuple[j][0], target_tuple[j][0])]
        for i in range(52):
            image_generator = ImageDataGenerator(rotation_range=179, horizontal_flip=True, vertical_flip=True)
            new_total_list_day3 += [(image_generator.random_transform(image_tuple[j][0], seed=None), target_tuple[j][0])]
    elif target_tuple[j][0] == 'trophozoite':
        new_total_list_day3 += [(image_tuple[j][0], target_tuple[j][0])]
        for i in range(5):
            image_generator = ImageDataGenerator(rotation_range=179, horizontal_flip=True, vertical_flip=True)
            new_total_list_day3 += [(image_generator.random_transform(image_tuple[j][0], seed=None), target_tuple[j][0])]

# Resize the image array to (32, 32, 3) or other specified sizes
res_day3 = np.zeros([len(new_total_list_day3), 32, 32, 3])

for i in range(len(new_total_list_day3)):
    res_day3[i] = cv2.resize(new_total_list_day3[i][0], (32, 32), interpolation=cv2.INTER_CUBIC)

# Get the x_total with shape as (n_total_image, 3072) or (n_total_image, other sizes)
x_total_day3 = res_day3.reshape(len(res_day3), -1)

# Get the target_total with shape as (n_total_image, 1)
target_total_day3 = np.zeros([len(new_total_list_day3), 1])
for i in range(len(new_total_list_day3)):
    if 'red blood cell' == new_total_list_day3[i][1]:
        target_total_day3[i] = 0
    elif 'ring' == new_total_list_day3[i][1]:
        target_total_day3[i] = 1
    elif 'schizont' == new_total_list_day3[i][1]:
        target_total_day3[i] = 2
    elif 'trophozoite' == new_total_list_day3[i][1]:
        target_total_day3[i] = 3

print(len(x_total_day3), len(target_total_day3))

# Save the x_total_day3 and target_total_day3 as np arrays with specified names
# np.save("/home/ubuntu/Exam1/x_total_image32three.npy", x_total_day3)
# np.save("/home/ubuntu/Exam1/target_total_image32three.npy", target_total_day3)


# Data of day6:
new_total_list_day6 = []
for j in range(len(image_tuple)):
    if target_tuple[j][0] == 'red blood cell':
        new_total_list_day6 += [(image_tuple[j][0], target_tuple[j][0])]
    elif target_tuple[j][0] == 'ring':
        new_total_list_day6 += [(image_tuple[j][0], target_tuple[j][0])]
        for i in range(18):
            image_generator = ImageDataGenerator(rotation_range=179, horizontal_flip=True, vertical_flip=True,
                                                 shear_range=0.6, featurewise_center=True,
                                                 featurewise_std_normalization=True)
            new_total_list_day6 += [(image_generator.random_transform(image_tuple[j][0], seed=None), target_tuple[j][0])]
    elif target_tuple[j][0] == 'schizont':
        new_total_list_day6 += [(image_tuple[j][0], target_tuple[j][0])]
        for i in range(52):
            image_generator = ImageDataGenerator(rotation_range=179, horizontal_flip=True, vertical_flip=True,
                                                 shear_range=0.6, featurewise_center=True,
                                                 featurewise_std_normalization=True)
            new_total_list_day6 += [(image_generator.random_transform(image_tuple[j][0], seed=None), target_tuple[j][0])]
    elif target_tuple[j][0] == 'trophozoite':
        new_total_list_day6 += [(image_tuple[j][0], target_tuple[j][0])]
        for i in range(5):
            image_generator = ImageDataGenerator(rotation_range=179, horizontal_flip=True, vertical_flip=True,
                                                 shear_range=0.6, featurewise_center=True,
                                                 featurewise_std_normalization=True)
            new_total_list_day6 += [(image_generator.random_transform(image_tuple[j][0], seed=None), target_tuple[j][0])]

# Resize the image array to (32, 32, 3) or other specified sizes
res_day6 = np.zeros([len(new_total_list_day6), 32, 32, 3])

for i in range(len(new_total_list_day6)):
    res_day6[i] = cv2.resize(new_total_list_day6[i][0], (32, 32), interpolation=cv2.INTER_CUBIC)

# Get the x_total with shape as (n_total_image, 3072) or (n_total_image, other sizes)
x_total_day6 = res_day6.reshape(len(res_day6), -1)

# Get the target_total with shape as (n_total_image, 1)
target_total_day6 = np.zeros([len(new_total_list_day6), 1])
for i in range(len(new_total_list_day6)):
    if 'red blood cell' == new_total_list_day6[i][1]:
        target_total_day6[i] = 0
    elif 'ring' == new_total_list_day6[i][1]:
        target_total_day6[i] = 1
    elif 'schizont' == new_total_list_day6[i][1]:
        target_total_day6[i] = 2
    elif 'trophozoite' == new_total_list_day6[i][1]:
        target_total_day6[i] = 3

print(len(x_total_day6), len(target_total_day6))

# Save the x_total_day6 and target_total_day6 as np arrays with specified names
# np.save("/home/ubuntu/Exam1/x_total_image32_six.npy", x_total_day6)
# np.save("/home/ubuntu/Exam1/target_total_image32_six.npy", target_total_day6)


# %% ---------------------------------------Set all random seed---------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# %% ----------------------------------- Specified all Parameters and paths --------------------------------------------
LR_day3 = 0.000035
LR_day6 = 0.000135
N_NEURONS_day3 = (1500, 800, 500, 100)
N_NEURONS_day6 = (2050, 850, 55)
N_EPOCHS_day3 = 1000
N_EPOCHS_day6 = 1000
BATCH_SIZE_day3 = 1500
BATCH_SIZE_day6 = 6500
DROPOUT_day3 = 0.1
DROPOUT_day6 = 0.05
cw = {0: 1.00,
      1: 1,
      2: 1,
      3: 1}

FILEPATH_day3 = '/home/ubuntu/Deep-Learning/Class Exercise/mlp_yongchaoqiao9day3.hdf5'
FILEPATH_day6 = '/home/ubuntu/Deep-Learning/Class Exercise/mlp_yongchaoqiao9day6.hdf5'

# #  Data for day3
# x_train_total_day3 = np.load("/home/ubuntu/Exam1/x_total_image32three.npy")
# target_train_total_day3 = np.load("/home/ubuntu/Exam1/target_total_image3three.npy")
x_train_total_day3 = x_total_day3
target_train_total_day3 = target_total_day3
print(x_train_total_day3.shape, target_train_total_day3.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train_total_day3, target_train_total_day3, random_state=SEED,
                                                    test_size=0.2, stratify=target_train_total_day3)
print(x_train.shape, x_test.shape)

x_train, x_test = x_train / 255, x_test / 255

y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
# MLP for day3:
# First layer
model_day3 = Sequential([Dense(N_NEURONS_day3[0], input_dim=3072, kernel_initializer=weight_init),
                    Activation('sigmoid'),
                    Dropout(DROPOUT_day3, seed=SEED),
                    BatchNormalization()])

# Hidden layers
for n_neurons in N_NEURONS_day3[1:]:
    model_day3.add(Dense(n_neurons, activation='relu', kernel_initializer=weight_init))
    model_day3.add(Dropout(DROPOUT_day3, seed=SEED))
    model_day3.add(BatchNormalization())

model_day3.add(Dense(4, activation="softmax", kernel_initializer=weight_init))

model_day3.compile(optimizer=Adam(lr=LR_day3), loss="categorical_crossentropy", metrics=['accuracy'])

# Show the summary
model_day3.summary()

# Save the best model on validation set
checkpointer_day3 = ModelCheckpoint(filepath=FILEPATH_day3, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train history
train_hist_day3 = model_day3.fit(x_train, y_train, batch_size=BATCH_SIZE_day3, epochs=N_EPOCHS_day3,
                       validation_data=(x_test, y_test),
                       callbacks=[checkpointer_day3], shuffle=True)

# # Data for day6
# x_train_total_day6 = np.load("/home/ubuntu/Exam1/x_total_image32_three_no_test.npy")
# target_train_total_day6 = np.load("/home/ubuntu/Exam1/target_total_image_32_three_no_test.npy")
x_train_total_day6 = x_total_day6
target_train_total_day6 = target_total_day6
print(x_train_total_day6.shape, target_train_total_day6.shape)

x_train_day6, x_test_day6, y_train_day6, y_test_day6 = train_test_split(
    x_train_total_day6, target_train_total_day6, random_state=42, test_size=0.2, stratify=target_train_total_day6)
print(x_train_day6.shape, x_test_day6.shape)

x_train_day6, x_test_day6 = x_train_day6 / 255, x_test_day6 / 255

y_train_day6, y_test_day6 = to_categorical(y_train_day6, num_classes=4), to_categorical(y_test_day6, num_classes=4)
# %% -------------------------------------- Training Prep ----------------------------------------------------------
# Day 6 :
# First layer
model_day6 = Sequential([Dense(N_NEURONS_day6[0], input_dim=3072, kernel_initializer=weight_init),
                    Activation('sigmoid'),
                    Dropout(DROPOUT_day6, seed=SEED),
                    BatchNormalization()])

# Hidden layers
for n_neurons in N_NEURONS_day6[1:]:
    model_day6.add(Dense(n_neurons, activation='relu', kernel_initializer=weight_init))
    model_day6.add(Dropout(DROPOUT_day6, seed=SEED))
    model_day6.add(BatchNormalization())

model_day6.add(Dense(4, activation="softmax", kernel_initializer=weight_init))

# Specified optimizer, loss, metrics
model_day6.compile(optimizer=Adam(lr=LR_day6), loss="categorical_crossentropy", metrics=['accuracy'])

# Show the summary
model_day6.summary()

# Add checkpoint to save the best model
checkpointer_day6 = ModelCheckpoint(filepath=FILEPATH_day6, monitor='val_loss', verbose=1, save_best_only=True,
                                    mode='min')

# Add LR reducing option
reducelr_day6 = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=30, verbose=1, mode='auto', min_delta=0.0001,
                             cooldown=0, min_lr=0)

# Train history
train_hist = model_day6.fit(x_train_day6, y_train_day6, batch_size=BATCH_SIZE_day6, epochs=N_EPOCHS_day6,
                       validation_data=(x_test_day6, y_test_day6), callbacks=[checkpointer_day6, reducelr_day6],
                            shuffle=True)

# Load the best weight
model_day6.load_weights(FILEPATH_day6)

# %% -------------------------------------- Test the ensemble model-----------------------------------------------------
#  Load the day1 test set
# x_total = np.load("/home/ubuntu/Exam1/x_total.npy")
# target_total = np.load("/home/ubuntu/Exam1/target_total.npy")

x_train_day1, x_test_day1, y_train_day1, y_test_day1 = train_test_split(x_total, target_total, random_state=42,
                                                                        test_size=0.2, stratify=target_total)
print(x_train_day1.shape, x_test_day1.shape)

x_train_day1, x_test_day1 = x_train_day1 / 255, x_test_day1 / 255

y_train_day1, y_test_day1 = to_categorical(y_train_day1, num_classes=4), to_categorical(y_test_day1, num_classes=4)



# Load high-score models
model_day3_best = load_model('mlp_yongchaoqiao9day3.hdf5')
model_day6_higher = load_model('mlp_yongchaoqiao9day6.hdf5')

# Ensemble models
y_pred_ensemble = np.argmax((model_day3_best.predict(x_test_day1) + model_day6_higher.predict(x_test_day1)) / 2, axis=1)

y_test_true_day1 = np.argmax(y_test_day1, axis=1)

# Test the ensemble model on the test set of day1
print("Cohen Kappa", cohen_kappa_score(y_pred_ensemble, y_test_true_day1))
print("F1 score", f1_score(y_pred_ensemble, y_test_true_day1, average='macro'))
print('Mean of these two score', 0.5 * (f1_score(y_pred_ensemble, y_test_true_day1, average='macro') +
                                        cohen_kappa_score(y_pred_ensemble, y_test_true_day1)))

