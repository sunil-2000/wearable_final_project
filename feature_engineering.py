import numpy as np


#################### convert data to matrix ####################################
mask_on_no_talk = open("properly_worn.csv")
mask_under_nose = open("below_nose(1).csv")
mask_on_no_talk_M = np.loadtxt(mask_on_no_talk, delimiter=",")
mask_under_nose_M = np.loadtxt(mask_under_nose, delimiter=",")
# print(mask_on_no_talk_M.shape, mask_under_nose_M.shape)
# print(type(mask_under_nose_M[0][0]))
################################################################################

################ feature engineering ###########################################
class Features:
    def __init__(self, csvFiles, windowSize):
        dataSetList = []
        for csv in csvFiles:
            opened_csv = open(csv)
            dataSetList.append(np.loadtxt(opened_csv, delimiter=","))

        dataSetList = map(lambda d: Features.set_labels(d), dataSetList)
        features_by_dataset = []
        for d in dataSetList:
            label = d[:, -1]
            datum = Features.add_additional_feature(d[:, :-1], label, windowSize)
            features_by_dataset.append(datum)

        self.features = np.concatenate(features_by_dataset,axis=0)
        self.full_dataset = Features.shuffle(self.features)
        self.xTr, self.yTr, self.xTe, self.yTe = Features.generate_train_test(
            self.full_dataset
        )

    @staticmethod
    def shuffle(data):
        """
        data is a full dataset with labels
        """
        full_dataset = np.concatenate(data)
        return np.random.shuffle(full_dataset)

    @staticmethod
    def generate_train_test(full_dataset):
        """
        generate training and testing dataset
        """
        train_size = int(0.8 * len(full_dataset))
        full_train = full_dataset[:train_size]
        xTr, yTr = full_train, full_train[:, -1]
        full_test = full_dataset[train_size:]
        xTe, yTe = full_test[:, :-1], full_test[:, -1]
        return xTr, yTr, xTe, yTe

    @staticmethod
    def set_labels(data):
        """
        ensure labels are -1 or +1 for svm
        """
        for i in range(len(data)):
            data[i, -1] = -1 if data[i, -1] == 0 else 1
        return data

    @staticmethod
    def add_additional_feature(data, label, windowSize):
        """
        add additional features to matrix
        """
        i, j = 0, windowSize
        mu_lst, std_lst, var_lst, maxi_lst, mini_lst = [], [], [], [], []
        min_max_diff_lst, median_lst = [], []
        count_above_mean_lst, count_below_mean_lst = [], []
        features = [
            mu_lst,
            std_lst,
            var_lst,
            maxi_lst,
            mini_lst,
            min_max_diff_lst,
            median_lst,
            count_above_mean_lst,
            count_below_mean_lst,
        ]
        while j + windowSize < len(data):
            dim = len(data[0])
            mu = [np.mean(data[i:j, k]) for k in range(dim)]
            std = [np.std(data[i:j, k]) for k in range(dim)]
            var = [np.var(data[i:j, k]) for k in range(dim)]
            maxi = [np.maximum(data[i:j, k]) for k in range(dim)]
            mini = [np.minimum(data[i:j, k]) for k in range(dim)]
            min_max_diff = [maxi[k] - mini[k] for k in range(dim)]
            median = [np.median(data[i:j, k]) for k in range(dim)]
            count_above_mean = [np.sum(data[k] > mu[k]) for k in range(dim)]
            count_below_mean = [np.sum(data[k] < mu[k]) for k in range(dim)]
            # append data
            values = [
                mu,
                std,
                var,
                maxi,
                mini,
                min_max_diff,
                median,
                count_above_mean,
                count_below_mean,
            ]
            for f in range(len(features)):
                features[f].append(values[f])
            i += windowSize
            j += windowSize
        features.append([label] * len(features[f]))  # append label as last column
        features = np.concatenate(features, axis=1)
        return features

    def print_statistics(self):
        print(
            "Size of training set : {}, size of testing set: {}".format(
                len(self.xTr), len(self.xTe)))
        print("\nTraining split:\n Count of +1 labels: {}, Count of -1 labels: {}".format(
        np.count_nonzero(self.yTr + 1), np.count_nonzero(self.yTr - 1)))
        print("\Testing split:\n Count of +1 labels: {}, Count of -1 labels: {}".format(
        np.count_nonzero(self.yTe + 1), np.count_nonzero(self.yTe - 1)))