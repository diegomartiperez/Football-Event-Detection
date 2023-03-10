import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
#import files
from models import *
from evaluate import *

#Data for frame features vs bounding box experiments
#labels = np.load('data/multiaction_recognition/frame_features/labels.npy')
#data = np.load('data/multiaction_recognition/frame_features/ff.npy')
#train_data, test_data,train_labels, test_labels = train_test_split(data,labels,test_size=0.20) #get train and test samples

#Data for Bounding box experiments, load train and test datasets
train_data = np.load('data/multiaction_recognition/bounding_boxes/train_4.npy')
train_labels = np.load('data/multiaction_recognition/bounding_boxes/labels_4.npy')
test_data = np.load('data/multiaction_recognition/bounding_boxes/test_4.npy')
test_labels = np.load('data/multiaction_recognition/bounding_boxes/test_labels_4.npy')
  
#class vocab for 10-class dataset or 4-class
#vocab = ['corner', 'forward-pass', 'freekick', 'high-pass', 'keeper-in-action', 'no-action', 'penalty', 'shot-at-goal', 'throw-in', 'tiki-taka']
vocab = ['high-pass', 'keeper-in-action', 'no-action','shot-at-goal'] 
label_processor = keras.layers.StringLookup(num_oov_indices=0,vocabulary=np.unique(vocab), mask_token=None)
class_vocab = label_processor.get_vocabulary()
classes = len(label_processor.get_vocabulary())
print("Number of classes:",classes)
print("Classes:",class_vocab)

def samples_to_augment(data,labels):
    indexes = []
    data_toaugment = []
    labels_toaugment = []
    for i in range(len(labels)):
        #if labels[i] != 2: #for the 4-class dataset
        if labels[i] != 5:#for the 10-class dataset      
            data_toaugment.append(data[i])
            labels_toaugment.append(labels[i])
    return data_toaugment, labels_toaugment

def augment(x_train):
    for sample in x_train:
        for frame in sample:
            i = 1 
            while i<len(frame):
                frame[i] = 1-frame[i] #change x-coordinate which equals to flipping the image
                i += 5
    return x_train

train_data = train_data.tolist()
train_labels = train_labels.tolist()

def augmentation(data,labels):
    x_to_aug,y_to_aug = samples_to_augment(data,labels) #get samples that will be augmented
    aug_train = augment(x_to_aug) #apply augmentation
    data = data + aug_train #add augmented samples to dataset
    labels = labels + y_to_aug #add augmented labels to dataset
    c = list(zip(data,labels)) #shuffle the data
    random.shuffle(c)
    data,labels = zip(*c)  
    return data,labels

#train_data,train_labels = augmentation(train_data,train_labels) #uncomment to apply data augmentation
#Get number of samples for each class
num_classes = {}
num_classes_test = {}
for i in range(classes):
    count = 1
    count_test = 1
    for sample in train_labels:
        if sample == i:
            num_classes[i] = count
            count += 1
    for sample in test_labels:
        if sample == i:
            num_classes_test[i] = count_test
            count_test += 1
print("Number of samples (train)",num_classes)
print("Number of samples (test)",num_classes_test)

x_train = np.array([np.array(item) for item in train_data])   
x_test = np.array([np.array(item) for item in test_data]) 

#Convert labels to categorical
y_train= to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Choose model
model = gru(x_train,classes) 
#model = lstm(x_train,classes) 
#model = transformer(x_train,classes) 

#choose class weights
#class_weights = {0: 2, 1: 1.5, 2: 2, 3: 1.25, 4: 1.2, 5: 0.6, 6: 3, 7: 1, 8: 2, 9: 2.5}
#class_weights = {0: 2, 1: 1, 2: 2, 3: 1, 4: 1.2, 5: 1, 6: 3, 7: 1, 8: 2, 9: 2.5}
class_weights = {0:1.2,1:1.4,2:0.7,3:1.4} 
print(class_weights)

#train model
batch = 32
history = model.fit(x_train,y_train,epochs=2,class_weight=class_weights,batch_size=batch,validation_split=0.15,verbose=1)
plots(history,"categorical_accuracy","val_categorical_accuracy") #show training plots
y_pred = model.predict(x_test) #run predictions

evaluate(y_pred,y_test.argmax(axis=1)) #compare predictions with actual classes

#Display Confusion Matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
#l = [0,1,2,3,4,5,6,7,8,9]
l = [0,1,2,3]
plot_cm(cm,l)
precision_recall(y_pred,y_test,len(class_vocab)) #plot precision-recall curve




