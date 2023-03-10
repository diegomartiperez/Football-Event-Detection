import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
#import files
from models import *
from evaluate import *
from display_videos import *

#Load data
train_data = np.load('data/shot_detection/x_train.npy')
train_labels = np.load('data/shot_detection/y_train.npy')
test_data = np.load('data/shot_detection/x_test.npy')
test_labels = np.load('data/shot_detection/y_test.npy')
#train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2]*train_data.shape[3])
#np.save('dataa/shot_detection/x_train2.npy', train_data)
#Get shot and no-action samples only
def reduce(data_csv,labels_csv):
    indexes = []
    for i in range(len(labels_csv)):
        if labels_csv[i] != "shot-at-goal" and labels_csv[i] != "no-action":
            indexes.append(i)
    for index in sorted(indexes, reverse=True):
        del labels_csv[index]
        del data_csv[index]

def reduce_test(test_labels):
    for i in range(len(test_labels)):
        if test_labels[i] != "shot-at-goal" and test_labels[i] != "no-action":
            test_labels[i] = "no-action"

train_data = train_data.tolist()
train_labels = train_labels.tolist()
test_data = test_data.tolist()
test_labels = test_labels.tolist()

#reduce dataset to only shot and no-action samples
reduce(train_data,train_labels)
reduce_test(test_labels)

#Get class vocabulary and convert labels to integers
label_processor = keras.layers.StringLookup(num_oov_indices=0,vocabulary=np.unique(train_labels), mask_token=None)
train_labels_int = label_processor(train_labels).numpy()
test_labels_int = label_processor(test_labels).numpy()
class_vocab = label_processor.get_vocabulary()
classes = len(label_processor.get_vocabulary())
print("Number of classes:",classes)
print("Classes:",class_vocab)

#get ID of no-action
index = 0
for i in class_vocab:
    if i == "no-action":
        no_act_id = index
    index += 1

#Get number of samples for each class
num_classes = {}
num_classes_test = {}
for i in range(classes):
    count = 1
    count_test = 1
    for sample in train_labels_int:
        if sample == i:
            num_classes[i] = count
            count += 1
    for sample in test_labels_int:
        if sample == i:
            num_classes_test[i] = count_test
            count_test += 1
print("Number of samples (train)",num_classes)
print("Number of samples (test",num_classes_test)

x_train = np.array([np.array(item) for item in train_data])
x_test = np.array([np.array(item) for item in test_data])

#Convert labels to categorical
y_train= to_categorical(train_labels_int)
y_test = to_categorical(test_labels_int,num_classes=2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = gru(x_train,classes)

#Set-up class weights
class_weights = {0:1,1:1.2} 
print(class_weights)

#Train model
batch = 32
history = model.fit(x_train,y_train,epochs=2,class_weight=class_weights,batch_size=batch,validation_split=0.15,verbose=1)
plots(history,"categorical_accuracy","val_categorical_accuracy")
y_pred = model.predict(x_test) #get predictions

threshold = 0.5 #set threshold value
cm,preds = evaluate_shot(no_act_id,y_pred,y_test.argmax(axis=1),class_vocab,threshold) #evaluate results

#Display Confusion Matrix
l = ["No-action","Shot"]
plot_cm(cm,l)

precision_recall(y_pred,y_test,len(class_vocab)) #plot precision-recall curve
plot_roc(y_pred,y_test) #plot ROC curve

#Display videos with predictions
test_videos = np.load('data/shot_detection/videos.npy') #load information about the videos to display them
videos_folder = "videos/"
dst_folder = "test_visualization/"
videos = [item[0] for item in test_videos]
index = [item[1] for item in test_videos]
for video in test_videos:
    vid(videos_folder,dst_folder,video[0],preds,y_pred,int(video[1]),threshold)

