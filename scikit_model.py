from math import inf
import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from feature_engineering import Features
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

################################################################################


class Model:
    def __init__(self, csv_list, windowSize):
        self.features = Features(csv_list, windowSize)
        self.X = self.features.X
        self.Y = self.features.Y
        self.knn, self.svm, self.nb, self.rf = None, None, None, None

    def create_knn_model(self, n_neighbors):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, p=2)
        xTr, xTe, yTr, yTe = train_test_split(
            self.X, self.Y, random_state=42, stratify=self.Y)
        model_out = model.fit(xTr, yTr)
        train_preds = model.predict(xTr)
        test_preds = model.predict(xTe)

        print("Total labels in train:{}, +1 labels:{}, -1 label:{}".format(len(yTr),
              np.count_nonzero(yTr[yTr > 0]), np.count_nonzero(yTr[yTr < 0])))
        print("Total labels in test:{}, +1 labels:{}, -1 label:{}".format(len(yTe),
              np.count_nonzero(yTe[yTe > 0]), np.count_nonzero(yTe[yTe < 0])))
        Model.model_accuracy(yTr, train_preds, yTe, test_preds, "knn", True)
        self.knn = model_out
        return model_out

    def create_svm_model(self):
        model = svm.SVC(kernel="polynomial")
        xTr, xTe, yTr, yTe = train_test_split(
            self.X, self.Y, random_state=42, stratify=self.Y)
        model_out = model.fit(xTr, yTr)
        train_preds = model.predict(xTr)
        test_preds = model.predict(xTe)

        print("Total labels in train:{}, +1 labels:{}, -1 label:{}".format(len(yTr),
              np.count_nonzero(yTr[yTr > 0]), np.count_nonzero(yTr[yTr < 0])))
        print("Total labels in test:{}, +1 labels:{}, -1 label:{}".format(len(yTe),
              np.count_nonzero(yTe[yTe > 0]), np.count_nonzero(yTe[yTe < 0])))
        Model.model_accuracy(yTr, train_preds, yTe, test_preds, "svm", True)
        self.svm = model_out
        return model_out
    
    def create_nb(self):
        model = GaussianNB()
        xTr, xTe, yTr, yTe = train_test_split(self.X, self.Y, random_state=42, stratify=self.Y)
        model_out = model.fit(xTr, yTr)
        train_preds = model.predict(xTr)
        test_preds = model.predict(xTe)
        print("Total labels in train:{}, +1 labels:{}, -1 label:{}".format(len(yTr),
              np.count_nonzero(yTr[yTr > 0]), np.count_nonzero(yTr[yTr < 0])))
        print("Total labels in test:{}, +1 labels:{}, -1 label:{}".format(len(yTe),
              np.count_nonzero(yTe[yTe > 0]), np.count_nonzero(yTe[yTe < 0])))
        Model.model_accuracy(yTr, train_preds, yTe, test_preds, "naive bayes", True)
        self.nb = model_out
        return model_out 
    
    def create_rf(self):
        model = RandomForestClassifier(random_state=42)
        xTr, xTe, yTr, yTe = train_test_split(self.X, self.Y, random_state=42, stratify=self.Y)
        model_out = model.fit(xTr, yTr)
        train_preds = model.predict(xTr)
        test_preds = model.predict(xTe)
        print("Total labels in train:{}, +1 labels:{}, -1 label:{}".format(len(yTr),
              np.count_nonzero(yTr[yTr > 0]), np.count_nonzero(yTr[yTr < 0])))
        print("Total labels in test:{}, +1 labels:{}, -1 label:{}".format(len(yTe),
              np.count_nonzero(yTe[yTe > 0]), np.count_nonzero(yTe[yTe < 0])))
        Model.model_accuracy(yTr, train_preds, yTe, test_preds, "random forest", True)
        self.rf = model_out
        return model_out 


    def k_fold_validation(self, model, model_name):
        kf = StratifiedKFold(shuffle=True)
        kfold_validated = model
        scores = cross_val_score(
            kfold_validated, self.X, self.Y, scoring='accuracy', cv=kf, n_jobs=-1)
        print("accuracy scores: {}".format(
            list(map(lambda x: round(x*100, 2), scores))))
        print("\nmean accuracy: {}%".format(round(np.mean(scores * 100), 2)))
        # stratify parameter ===> each set sample from same distribution

    @staticmethod
    def model_accuracy(yTr, train_preds, yTe, test_preds, model_type, print_on):
        tr_acc = sklearn.metrics.accuracy_score(yTr, train_preds)
        te_acc = sklearn.metrics.accuracy_score(yTe, test_preds)
        if print_on:
            print("Training accuracy ({}): {}%".format(
                model_type, round(tr_acc * 100, 2)))
            print("Testing accuracy ({}): {}%".format(
                model_type, round(te_acc * 100, 2)))
        return tr_acc, te_acc

    def knn_k_plot(self):
        # PLOT to determine optimal 'k' hyperparameter
        k_lst = []
        xTr, xTe, yTr, yTe = train_test_split(
            self.X, self.Y, random_state=42)  # 50% of each label per set
        for k in range(1, 100):
            neigh = KNeighborsClassifier(
                n_neighbors=k, p=2)  # euclidean distance
            neigh.fit(self.features.xTr, self.features.yTr)
            # print("k={}\n".format(k))
            knn_train_preds = neigh.predict(self.features.xTr)
            knn_test_preds = neigh.predict(self.features.xTe)
            tr_acc, te_acc = self.model_accuracy(
                self.features.yTr, knn_train_preds, self.features.yTe, knn_test_preds, "KNN", False
            )
            k_lst.append({"k": k, "tr_acc": tr_acc, "te_acc": te_acc})

        k_s = [elt["k"] for elt in k_lst]
        k_err = [1 - elt["te_acc"] for elt in k_lst]
        plt.plot(k_s, k_err)
        plt.xlabel("k")
        plt.ylabel("test error")
        plt.show()
