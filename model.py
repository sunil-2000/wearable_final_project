from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# import visclassifier

#################### convert data to matrix ####################################
mask_on_no_talk = open("properly_worn.csv")
mask_under_nose = open("below_nose(1).csv")
mask_on_no_talk_M = np.loadtxt(mask_on_no_talk, delimiter=",")
mask_under_nose_M = np.loadtxt(mask_under_nose, delimiter=",")
# print(mask_on_no_talk_M.shape, mask_under_nose_M.shape)
# print(type(mask_under_nose_M[0][0]))
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
def set_labels(data):
    """
    ensure labels are -1 or +1 for svm
    """
    for i in range(len(data)):
        data[i, -1] = -1 if data[i, -1] == 0 else 1
    return data


full_dataset = set_labels(full_dataset)


train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

xTr = torch.from_numpy(full_dataset[:train_size])
xTe = torch.from_numpy(full_dataset[train_size:])
################################################################################


class SVM(nn.Module):
    def __init__(self, dim):
        super(SVM, self).__init__()
        self.w = nn.Parameter(torch.zeros(dim, 1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        # print("forward result")
        # res = x.float() @ self.w + self.b
        # print(res.shape)
        # print(res)
        # exit
        return (x.float() @ self.w + self.b).reshape(x.shape[0])


def hinge_loss(y_pred, y_true):
    loss = torch.sum(torch.clamp(input=1 - y_pred * y_true, min=0))
    return loss / len(y_pred)


def primalSVM(x, y, num_epochs=10000, C=1):
    d = x.shape[1]
    svmclassify = SVM(d)
    optimizer = optim.SGD(svmclassify.parameters(), lr=1e-1)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = svmclassify.forward(x)
        h_loss = hinge_loss(pred, y) * C + torch.norm(svmclassify.w)
        h_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print("epoch {} loss {}".format(epoch + 1, h_loss.item()))
            # print("\n pred shape:{}, yTr shape:{}".format(pred.shape, yTr.shape))

    return svmclassify


# testing


yTr = xTr[:, -1]
xTrFeatures = xTr[:, :-1]
xTeFeatures = xTe[:, :-1]
yTe = xTe[:, -1]
fun = primalSVM(xTrFeatures, yTr, C=10)
# visclassifier.visclassifier(fun, xTr, yTr)
err = torch.mean((torch.sign(fun(xTrFeatures)) != yTr).float())
te_err = torch.mean((torch.sign(fun(xTeFeatures)) != yTe).float())
print("Training error: %2.1f%%" % (err * 100))
print("Testing error: %2.1f%%" % (te_err * 100))
