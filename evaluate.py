import numpy as np
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from pyllr.pav_rocch import PAV, ROCCH
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score

def evaluate(y_pred,y_test):    
    right = 0
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == y_test[i]:            
            right += 1
    print("Right:",right)
    print("All:",len(y_pred))
    acc = right/len(y_pred)
    print("Accuracy:",acc)

def evaluate_shot(no_act_id,y_pred,y_test,class_vocab,threshold):  
    results = {} 
    preds = []
    for i in range(len(y_pred)): #set prediction to 0 if lower than threshold and 1 if above threshold
        if y_pred[i][1]<threshold:
            preds.append(0)
        else:
            preds.append(1)
    for i in range(len(class_vocab)):
        right = 0
        total = 0
        if i != no_act_id: #50 frame confidence for shots.
            # If shot is detected 50 frames before it is considered as correct prediction
            for j in range(len(preds)):
                if y_test[j] == i:
                    try:
                        if y_test[j] == preds[j] or y_test[j] == preds[j-1] or y_test[j] == preds[j+1]:
                            right += 1
                        total += 1
                    except:
                        if y_test[j] == preds[j] or y_test[j] == preds[j-1]:
                            right += 1
                        total += 1          
        else:
            for j in range(len(y_pred)):
                if y_test[j] == i:
                    try:
                        if y_test[j] == preds[j] or y_test[j-1] == preds[j] or y_test[j+1] == preds[j]:
                            right += 1
                        total += 1
                    except:
                        if y_test[j] == preds[j] or y_test[j] == preds[j]:                          
                            right += 1
                        total += 1
        results[str(i)] = right,total
    cm = np.array([[results["0"][0],results["0"][1]-results["0"][0]],[results["1"][1]-results["1"][0],results["1"][0]]])
    return cm,preds

def plots(history,acc,val):
    plt.plot(history.history[acc])
    plt.plot(history.history[val])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train',"val"], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train',"val"], loc='upper left')
    plt.show()

def plot_cm(cm,l):
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=l)
    disp1.plot(cmap=plt.cm.Blues)
    plt.show()

# For each class
def precision_recall(y_score,Y_test,n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    print("AP:",average_precision)

    display = PrecisionRecallDisplay(recall=recall["micro"],precision=precision["micro"],
    average_precision=average_precision["micro"],)
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")
    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    _, ax = plt.subplots(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    plt.show()

def plot_roc(y_pred,y_test):
    shots = [item[1] for item in y_pred]
    shots = np.array(shots)
    pav = PAV(shots,y_test.argmax(axis=1))
    rocch = ROCCH(pav)
    fig, ax = plt.subplots()
    pmiss,pfa = rocch.Pmiss_Pfa()
    ax.plot(pfa,pmiss,label='rocch')
    plt.ylabel('False Negatives')
    plt.xlabel('False Positives')
    ax.grid()
    ax.set_title("ROC convex hull")
    ax.legend(loc='best', frameon=False)
    plt.show()