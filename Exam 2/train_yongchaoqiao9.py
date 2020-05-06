# # %%---------------------------------------Introduction---------------------------------------------------------------
# # Generally, Day1 is used to prepare the dataset, adjust the parameters of the CNN to get a high test score, and test
# the prediction function that will be submitted on the blackboard.
# Specifically, there are 8 steps in day1:
# 1. Download the dataset to the cloud with the given link.
# 2. Get the png and txt files from the train.zip file.
# 3. Read and match the information of the images and contents in the txt files.
# 4. Augment the original dataset to 9290 total observations and decide the resized size by checking the size
# distribution of the augmented dataset.
# 5. Prepare the x_total_120_160 and target_total_120_160 dataset
# 6. Find the distribution of the categories for the augmented dataset and generate the target_category_coding_120_160
# which wil be used as the strategy to split the augmented dataset to get train and test sets.
# 7. Build the CNN network, tune the LR, N_NEURONS, N_EPOCHS, BATCH_SIZE and DROPOUT. Use different optimizers, weight
# initialization methods, learning rate schedulers
# 8. Get the best model for different parameters and write the predict_yongchaoqiao9.py.

# Note:
# 1. To keep more information of the original dataset, I resize the original images into (120, 160, 3). This also takes
# the memory of CPUs and GPU into consideration.
# 2. Since the original dataset contains 40 categories and some categories only show one time which will be hard to
# split and get train and test sets with the same distribution, I augmented the original data to 10 times of the number
# of the original observations
# 3. Then randomly split the x_total_120_160 and target_total_120_160 into x_train & y_train and x_test & y_test with
# seed=143 and strategy as target_category_coding_120_160.
# 4. I tried binary output and probability output in train loss and test loss respectively. And finally I use
# probability output for train loss and  test loss without specified weight for different positions to get my
# best models

# # %%---------------------------------------Import packages------------------------------------------------------------
import os
import re
import cv2
import torch
import zipfile
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# # %%---------------------------------------Data Preparation-----------------------------------------------------------
# Step 1:
# Download the data into the cloud
if "train-Exam2.zip" not in os.listdir(os.getcwd()):
    try:
        os.system(
            "wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    except:
        print("There as a problem downloading the data!")
        raise
    if "train-Exam2.zip" not in os.listdir(os.getcwd()):
        print("There as a problem downloading the data!")

# Step 2:
# Get the contents from the train.zip file
z = zipfile.ZipFile(r'train-Exam2.zip', 'r')

for f in z.namelist():
    item = z.open(f, 'r')
    z.extract(f, r'data')
z.close()

# Step 3:
# Read images with corresponding categories with read_directory function
train_cell_dir = "data/train"
train_name = os.listdir(train_cell_dir)
print(len(train_name))


def read_directory(directory_name):
    image_id = []
    content_id = []
    for filename in sorted(os.listdir(directory_name)):
        if re.findall(r"\.png", filename):
            img = cv2.imread(directory_name + "/" + filename)
            image_id += [(img, filename.strip(r"\.png"))]
        elif re.findall(r"\.txt", filename):
            txtf = open(directory_name + "/" + filename)
            content = txtf.read().splitlines()
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

# Step 4:
# Augment the original dataset
new_total_list = []
for j in range(len(image_tuple)):
    new_total_list += [(image_tuple[j][0], target_tuple[j][0])]
    print(j)
    for i in range(9):
        image_generator = ImageDataGenerator(rotation_range=179, horizontal_flip=True, vertical_flip=True,
                                             shear_range=0.6, featurewise_center=True,
                                             featurewise_std_normalization=True, fill_mode='reflect')
        new_total_list += [(image_generator.random_transform(image_tuple[j][0], seed=None), target_tuple[j][0])]

# Check the distribution of the original picture size
ctw = 0
cth = 0
for i in range(len(new_total_list)):
    # print(image_tuple[i][0].shape)
    if new_total_list[i][0].shape == (1200, 1600, 3):
        ctw += 1
    elif new_total_list[i][0].shape == (1383, 1944, 3):
        cth += 1
    print(new_total_list[i][0].shape)
print(ctw, cth, ctw + cth)

# Step 5:
# Resize the image array to (120, 160, 3)
res = np.zeros([len(new_total_list), 120, 160, 3])

for i in range(len(new_total_list)):
    res[i] = cv2.resize(new_total_list[i][0], (160, 120), interpolation=cv2.INTER_CUBIC)
    # res[i] = np.clip(res[i] / 255, 0, 1)
    # imshow(res[i]/255)
    # plt.show()

# Get the target_total with shape as (n_image, 7)
target_total = np.zeros([len(new_total_list), 7])
for i in range(len(new_total_list)):
    if r"red blood cell" in new_total_list[i][1]:
        target_total[i][0] = 1
    if r"difficult" in new_total_list[i][1]:
        target_total[i][1] = 1
    if r"gametocyte" in new_total_list[i][1]:
        target_total[i][2] = 1
    if r"trophozoite" in new_total_list[i][1]:
        target_total[i][3] = 1
    if r"ring" in new_total_list[i][1]:
        target_total[i][4] = 1
    if r"schizont" in new_total_list[i][1]:
        target_total[i][5] = 1
    if r"leukocyte" in new_total_list[i][1]:
        target_total[i][6] = 1

# Check the match the elements of target_total and new_total_list
for j in range(10):
    print(target_total[j], new_total_list[j][1])

# Check all  frequency of each single position
All_frequency = target_total.sum(axis=0)
print(All_frequency)

# Step 6:
# Find the number of  categories of the images in the augmented dataset
dic = {}
for i in range(len(new_total_list)):
    if dic.get(str(sorted(new_total_list[i][1]))) is None:
        dic.update({str(sorted(new_total_list[i][1])): [1]})
    elif dic.get(str(sorted(new_total_list[i][1]))) is not None:
        dic[str(sorted(new_total_list[i][1]))] += [1]

sorted_dict = sorted(dic.items(), key=lambda x: sum(x[1]), reverse=True)
print(len(dic.keys()))

# Find the number of images in each category
'''for key, value in dic.items():
    print(key, sum(value))
for i in range(len(sorted_dict)):
    print(sorted_dict[i][0])'''

# Give each image a label (from 0 to 39) based on the frequency of its category in a descending order
# This label will be used to split the augmented dataset into train and test sets.
target_category_coding = np.zeros([len(new_total_list), 1])
for i in range(len(new_total_list)):
    for j in range(len(sorted_dict)):
        if str(sorted(new_total_list[i][1])) == sorted_dict[j][0]:
            target_category_coding[i] = j
for j in range(40):
    print(sorted_dict[int(target_category_coding[j])][0], '\n', sorted(new_total_list[j][1]))
print(target_category_coding.shape, res.shape, target_total.shape)

# Save the x_total_120_160, target_category_coding_120_160 and target_total_120_160 as np arrays
np.save("target_category_coding_120_160.npy", target_category_coding)
np.save("x_total_120_160.npy", res)
np.save("target_total_120_160.npy", target_total)

# Step 7:
# Build and train the model
# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0")
SEED = 413
torch.manual_seed(413)
torch.cuda.manual_seed_all(413)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.1
N_EPOCHS = 3000
BATCH_SIZE = 412
DROPOUT1 = 0.7
DROPOUT2 = 0.4
patience = 500


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (5, 5))  # output (n_examples, 8, 116, 156)
        self.convnorm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 8, 58, 78)
        self.conv2 = nn.Conv2d(8, 16, (3, 3))  # output (n_examples, 16, 56, 76)
        self.convnorm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 28, 38)
        self.conv3 = nn.Conv2d(16, 32, (3, 3))  # output (n_examples, 32, 26, 36)
        self.convnorm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 13, 18)
        self.conv4 = nn.Conv2d(32, 48, (3, 3))  # output (n_examples, 48, 11, 16)
        self.relectpad = nn.ReflectionPad2d((0, 0, 1, 0))  # 13 18
        self.convnorm4 = nn.BatchNorm2d(48)
        self.pool4 = nn.MaxPool2d((2, 2))  # output (n_examples, 48, 6, 9)
        self.linear1 = nn.Linear(48 * 6 * 8, 128)  # input will be flattened to (n_examples, 48*6*8)
        self.linear1_bn = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(DROPOUT1)
        self.linear2 = nn.Linear(128, 64)
        self.linear2_bn = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(DROPOUT2)
        self.linear3 = nn.Linear(64, 7)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x.permute(0, 3, 1, 2)))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.act(self.conv3(x))))
        x = self.pool4(self.convnorm4(self.act(self.relectpad(self.conv4(x)))))
        x = self.drop1(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.drop2(self.linear2_bn(self.act(self.linear2(x))))
        return self.linear3(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.normal_(m.bias.data, 0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.normal_(m.bias.data, 0, 0.02)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_total = np.load("x_total_120_160.npy")
target_total = np.load("target_total_120_160.npy")
target_category_coding = np.load("target_category_coding_120_160.npy")
print(x_total.shape, target_total.shape)
x_train, x_test, y_train, y_test = train_test_split(x_total, target_total, random_state=SEED, test_size=0.2,
                                                    stratify=target_category_coding)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# Reshapes to (n_examples, n_channels, height_pixels, width_pixels)
x_train, x_test = x_train / 255, x_test / 255
All_frequency = y_train.sum(axis=0)
print(All_frequency)
np.save("x_test_120_160.npy", x_test)
np.save("y_test_120_160.npy", y_test)
x_train, y_train = torch.tensor(x_train).float().to(device), torch.tensor(y_train).to(device)
x_train.requires_grad = True
y_train.requires_grad = False
x_test, y_test = torch.tensor(x_test).float().to(device), torch.tensor(y_test).to(device)
x_test.requires_grad = False
y_test.requires_grad = False
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
# Load the model to GPU
model = CNN().to(device)

# Initialize the weights
model.apply(weights_init)

# Specify optimizers
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Specify learning rate schedulers
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30, verbose=True, min_lr=8e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=5e-7)

# Specify different criterion
criterion1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array([1, (929 - 255) / 255, (929 - 101) / 101,
                                                                    (929 - 462) / 462, (929 - 204) / 204,
                                                                    (929 - 125) / 125, (929 - 59) / 59]),
                                                          device=device))
criterion2 = nn.BCEWithLogitsLoss()
criterion3 = nn.BCELoss()
best_model_loss = None
counter = 0
PATH = 'model_yongchaoqiao9day4.pt'
# PATH = 'model_yongchaoqiao9day6.pt'
# PATH = 'model_yongchaoqiao9day71.pt'
# PATH = 'model_yongchaoqiao9day72.pt'
# PATH = 'model_yongchaoqiao9day73.pt'

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    for batch in range(len(x_train) // BATCH_SIZE + 1):
        # index = [i for i in range(batch * BATCH_SIZE, np.min(((batch + 1) * BATCH_SIZE, len(x_train))))]
        index = [i for i in range(0, len(x_train))]
        np.random.shuffle(index)
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[index][inds])
        loss = criterion2(logits, y_train[index][inds])
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_train += loss.item()
    # Calculate test loss
    model.eval()
    with torch.no_grad():
        # y_test_pred = torch.sigmoid(model(x_test))。
        y_test_pred = model(x_test)
        loss = criterion2(y_test_pred, y_test.float())
        loss_test = loss.item()
        # scheduler.step(loss_test)
    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(epoch, loss_train / (len(x_train) // BATCH_SIZE + 1),
                                                                   loss_test))

    # Early stopping
    if best_model_loss is None:
        best_model_loss = loss_test
        print(f'Validation loss decreased ({np.Inf:.6f} --> {best_model_loss:.6f}).  Saving model as ' + '\"' + PATH +
              '\"')
        torch.save(model.state_dict(), PATH)
    elif loss_test < best_model_loss:
        print(f'Validation loss decreased ({best_model_loss:.6f} --> {loss_test:.6f}).  Saving model as ' + '\"' + PATH
              + '\"')
        best_model_loss = loss_test
        torch.save(model.state_dict(), PATH)
        counter = 0
    else:
        counter += 1
        print(f'EarlyStopping counter: {counter} out of {patience}')
        if counter >= patience:
            print("Early stopping")
            break

# Test the saved best model
x_test_cpu = x_test.to(torch.device('cpu'))
y_test_cpu = y_test.to(torch.device('cpu'))
modelt = CNN().to(torch.device('cpu'))
modelt.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
modelt.eval()
criterion = nn.BCEWithLogitsLoss()
print('Validation Loss: ', criterion(modelt(x_test_cpu), y_test_cpu))

# After about hundred times' tries, I found some good models. Fore example, one of them is:
# # LR = 0.1, N_EPOCHS = 3000, BATCH_SIZE = 412, DROPOUT1 = 0.7, DROPOUT2 = 0.4 the patience for early stopping is 500,
# the CNN structure has shown above with TMax in CosineAnnealingLR as 100. The optimizer is Adam, the initialization
# method is xavier_normal_. Then I combined these models with the models on day4, day5 and day6 to find the
# best ensemble model. Finally I got my best model: three are today's good single models, one is the best single model
# on day6 and the rest one is the best model on day4.

