
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay

class Classifier:
    def __init__(self):
        pass

    def plot_distribution(self, data, feature):
        feature = feature - 1
        plt.figure()
        seaborn.displot(data=data, x=feature, hue=(len(data.columns)-1))

    def plot_scatterplot(self, feature_a, feature_b):
        feature_a = feature_a - 1
        feature_b = feature_b - 1
        plt.figure()
        seaborn.scatterplot(data=data, x=feature_a, y=feature_b, hue=(len(data.columns)-1))

    def get_confusion_matrix_result(self, target, target_pred, title):
        cm = confusion_matrix(target, target_pred)
        TN, FP, FN, TP = cm.ravel()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        disp.ax_.set_title("Confusion Matrix {}".format(title))
        #accuracy
        ACC = (TP+TN) / (TP+TN+FN+FP)
        #precision
        PRE = TP / (TP+FP)
        #sensitivity
        Sn = TP / (TP+FN)
        #specificity
        Sp = TN / (TN+FP)
        #F2 Score
        F2 = (5 * PRE * Sn) / (4 * PRE + Sn)

        print("Accuracy = {}, Precision = {}, Sensitivity = {}, Specificity = {}, F2 Score = {}".format(ACC, PRE, Sn, Sp, F2))

    def roc_plot(self, target, target_pred):
        fpr, tpr, thresholds = roc_curve(target, target_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc, )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")

    def precision_recall_curve_plot(self,target ,target_pred):
        precision, recall, thresholds = precision_recall_curve(target, target_pred, pos_label=1)
        precision_inv = precision[::-1]
        recall_inv = recall[::-1]
        pr_res = np.interp(0.5, recall_inv, precision_inv)
        print("PR Score at recall of 50 is:" + str(pr_res))

        plt.figure()
        plt.axvline(0.5, 0, color="black", linestyle="dotted", label="Recall=0.5")
        plt.axhline(pr_res, 0, color="black", linestyle="dotted",
                    label=f"Precision={pr_res}")
        lw = 2
        plt.plot(recall, precision, color="darkorange", lw=lw, label="Precision Recall Curve", )
        plt.plot([0, 1], [0.5, 0.5], color="navy", lw=lw, linestyle="--")
        plt.plot(0.5, pr_res, marker='x', color="red")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        plt.legend(loc="lower right")

    def summary(self):
        self.model.summary()

    def predict(self, data):
        preds = self.model.predict(data)
        return preds

    def load(self, path='parameter.h5'):
        self.model = load_model(path)
        self.summary()

    def evaluate(self, test_data, test_target):
        test_loss, test_accuracy = self.model.evaluate(test_data, test_target)
        return test_loss, test_accuracy