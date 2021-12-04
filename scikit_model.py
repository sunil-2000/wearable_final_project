from math import inf
import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import matplotlib.pyplot as plt

#################### convert data to matrix ####################################
mask_on_no_talk = open("properly_worn.csv")
mask_under_nose = open("below_nose(1).csv")
mask_on_no_talk_M = np.loadtxt(mask_on_no_talk, delimiter=",")
mask_under_nose_M = np.loadtxt(mask_under_nose, delimiter=",")
# print(mask_on_no_talk_M.shape, mask_under_nose_M.shape)
# print(type(mask_under_nose_M[0][0]))
################################################################################

################ feature engineering ###########################################
print(
    "Size of +1 lables (mask not worn properly) : {}, \
    size of -1 labels mask worn properly:{}".format(
        len(mask_under_nose_M), len(mask_on_no_talk_M)
    )
)


def calculate_magnitude(vector):
    """
    magnitude of vector
    """
    mag = 0
    for elt in vector:
        mag += elt ** 2
    return math.sqrt(mag)


# change 0 label to -1
def set_labels(data):
    """
    ensure labels are -1 or +1 for svm
    """
    for i in range(len(data)):
        data[i, -1] = -1 if data[i, -1] == 0 else 1
    return data


def add_additional_feature(data):
    """
    add additional features to matrix
    """
    magnitudes = []  # magnitude feature
    for i in range(len(data)):
        mag = calculate_magnitude(data[i, :-1])  # exclude label
        magnitudes.append([mag])
    newdata = np.append(data, magnitudes, axis=1)
    return newdata


set_labels(mask_on_no_talk_M)
set_labels(mask_under_nose_M)
mask_on_no_talk_M_data = mask_on_no_talk_M[:, :-1]
# mask_on_no_talk_M_data = add_additional_feature(mask_on_no_talk_M[:, :-1])
mask_on_no_talk_M_labels = mask_on_no_talk_M[:, -1]

mask_under_nose_M_data = mask_under_nose_M[:, :-1]
# mask_under_nose_M_data = add_additional_feature(mask_under_nose_M[:, :-1])
mask_under_nose_M_labels = mask_under_nose_M[:, -1]


def align_data(data, labels):
    """
    add labels column back to data
    """
    labels = np.reshape(labels, (labels.shape[0], 1))
    complete_dataset = np.append(data, labels, axis=1)
    return complete_dataset


mask_on_no_talk_M = align_data(mask_on_no_talk_M_data, mask_on_no_talk_M_labels)
mask_under_nose_M = align_data(mask_under_nose_M_data, mask_under_nose_M_labels)
print(
    "shape of mask_on:{}, shape of mask_under_nose:{}".format(
        mask_on_no_talk_M.shape, mask_under_nose_M.shape
    )
)
################################################################################

################ create test, train datasets ###################################
full_dataset = np.concatenate((mask_on_no_talk_M, mask_under_nose_M))
np.random.shuffle(full_dataset)

# normalize data
min_max_rescale = lambda x, mini, maxi: (x - mini) / (maxi - mini)
means, std = [], []
for i in range(full_dataset.shape[1] - 1):  # dim -1 bc label is last col
    means.append(np.mean(full_dataset[i, :]))
    std.append(np.std(full_dataset[i, :]))
    minimum = np.amin(full_dataset[:, i])
    maximum = np.amax(full_dataset[:, i])
    full_dataset[:, i] = np.array(
        [min_max_rescale(x, minimum, maximum) for x in full_dataset[:, i]]
    )

# print(full_dataset[0])
# min max rescaling

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

xTr = full_dataset[:train_size]
yTr = xTr[:, -1]
print(
    "Training split:\n Count of +1 labels: {}, Count of -1 labels: {}".format(
        np.count_nonzero(yTr + 1), np.count_nonzero(yTr - 1)
    )
)
xTrFeatures = xTr[:, :-1]

xTe = full_dataset[train_size:]
yTe = xTe[:, -1]
xTeFeatures = xTe[:, :-1]
################################################################################
CList = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
gammaList = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

errorMatrix = [[0 for i in CList] for g in gammaList]
bestC, bestGamma = 0, 0
cError, gammaError = float("inf"), float("inf")

# for i in range(len(CList)):
#     C = CList[i]
#     for j in range(len(gammaList)):
#         gamma = gammaList[j]
#         SVM = svm.SVC(kernel="rbf", C=C, gamma=gamma)
#         SVM.fit(xTrFeatures, yTr)
#         train_preds = SVM.predict(xTrFeatures)
#         error = sklearn.metrics.accuracy_score(yTr, train_preds)
#         print(i, j)
#         errorMatrix[i][j] = error

#         bestGamma = gamma if error < gammaError else bestGamma
#         gammaError = min(error, gammaError)

#     bestC = C if error < cError else bestC
#     cError = min(error, cError)

# print(bestC, bestGamma)


def model_accuracy(yTr, train_preds, yTe, test_preds, model_type, print_on):
    tr_acc = sklearn.metrics.accuracy_score(yTr, train_preds)
    te_acc = sklearn.metrics.accuracy_score(yTe, test_preds)
    if print_on:
        print("Training accuracy ({}): {}%".format(model_type, round(tr_acc * 100, 2)))
        print("Testing accuracy ({}): {}%".format(model_type, round(te_acc * 100, 2)))
    return tr_acc, te_acc


bestSVM = svm.SVC(kernel="rbf")
bestSVM.fit(xTrFeatures, yTr)
train_preds = bestSVM.predict(xTrFeatures)
test_preds = bestSVM.predict(xTeFeatures)
model_accuracy(yTr, train_preds, yTe, test_preds, "SVM", True)

################################################################################

############################### KNN Attempt ####################################
neigh = KNeighborsClassifier(n_neighbors=5, p=2)
neigh.fit(xTrFeatures, yTr)
# print(xTe.shape, yTe.shape)
# print(xTe[0])
# dummy_row = [242, 228, 2309, 300, 200, 1]
# xTe = np.append(xTe, [dummy_row[:-1]], axis=0)
# yTe = np.append(yTe, [dummy_row[-1]], axis=0)
knn_train_preds = neigh.predict(xTrFeatures)
knn_test_preds = neigh.predict(xTeFeatures)
model_accuracy(yTr, knn_train_preds, yTe, knn_test_preds, "KNN", True)

# PLOT to determine optimal 'k' hyperparameter
k_lst = []
for k in range(1, 100):
    neigh = KNeighborsClassifier(n_neighbors=k, p=2)  # euclidean distance
    neigh.fit(xTrFeatures, yTr)
    # print("k={}\n".format(k))
    knn_train_preds = neigh.predict(xTrFeatures)
    knn_test_preds = neigh.predict(xTeFeatures)
    tr_acc, te_acc = model_accuracy(
        yTr, knn_train_preds, yTe, knn_test_preds, "KNN", False
    )
    k_lst.append({"k": k, "tr_acc": tr_acc, "te_acc": te_acc})

k_s = [elt["k"] for elt in k_lst]
k_err = [1 - elt["te_acc"] for elt in k_lst]
plt.plot(k_s, k_err)
plt.xlabel("k")
plt.ylabel("test error")
plt.show()
