import numpy as np
################ feature engineering ###########################################


class Features:
    def __init__(self, csvFiles, windowSize):
        dataSetList = []
        print(csvFiles)
        for csv in csvFiles:
            opened_csv = open(csv)
            dataSetList.append(np.loadtxt(
                opened_csv, delimiter=","))

        dataSetList = map(lambda d: Features.set_labels(d), dataSetList)
        features_by_dataset = []
        for d in dataSetList:
            label = d[0, -1]  # all labels are same for given dataset
            print("d"+str(len(d)))
            datum = Features.gen_features(d[:, :-1], label, windowSize)
            features_by_dataset.append(datum)
            # print(features_by_dataset.shape)

        self.features = np.concatenate(features_by_dataset, axis=0)
        print(self.features.shape)
        self.full_dataset = Features.shuffle(self.features)
        print(self.full_dataset.shape)
        self.xTr, self.yTr, self.xTe, self.yTe = Features.generate_train_test(
            self.full_dataset
        )
        self.X, self.Y = self.full_dataset[:, :-1], self.full_dataset[:, -1]

    @ staticmethod
    def shuffle(data):
        """
        data is a full dataset with labels
        """
        np.random.shuffle(data)
        return data

    @ staticmethod
    def generate_train_test(full_dataset):
        """
        generate training and testing dataset
        """
        train_size = int(0.8 * len(full_dataset))
        full_train = full_dataset[:train_size]
        xTr, yTr = full_train[:, :-1], full_train[:, -1]
        full_test = full_dataset[train_size:]
        xTe, yTe = full_test[:, :-1], full_test[:, -1]
        return xTr, yTr, xTe, yTe

    @ staticmethod
    def set_labels(data):
        """
        ensure labels are -1 or +1 for svm
        """
        for i in range(len(data)):
            data[i, -1] = -1 if data[i, -1] <= 0 else 1
        return data

    @ staticmethod
    def gen_features(data, label, windowSize):
        """
        add additional features to matrix
        """
        i, j = 0, windowSize

        ftr_dim_size = 9*4
        features = np.zeros(37).reshape((1, ftr_dim_size + 1))
        iters = 0
        while j + windowSize < len(data):
            dim = len(data[0])
            # print("test:{}".format(len(data[i:j, 1])))
            mu = [np.mean(data[i:j, k]) for k in range(dim)]
            std = [np.std(data[i:j, k]) for k in range(dim)]
            var = [np.var(data[i:j, k]) for k in range(dim)]
            maxi = [max(data[i:j, k]) for k in range(dim)]
            mini = [min(data[i:j, k]) for k in range(dim)]
            min_max_diff = [maxi[k] - mini[k] for k in range(dim)]
            median = [np.median(data[i:j, k]) for k in range(dim)]
            count_above_mean = [np.count_nonzero(
                data[i:j, k] > mu[k]) for k in range(dim)]
            count_below_mean = [np.count_nonzero(
                data[i:j, k] < mu[k]) for k in range(dim)]
            # append data
            values = np.array([
                mu,
                std,
                var,
                maxi,
                mini,
                min_max_diff,
                median,
                count_above_mean,
                count_below_mean
            ])
            flattened_values = np.append(values.flatten().reshape(
                (1, ftr_dim_size)), label).reshape((1, ftr_dim_size+1))

            features = np.concatenate((features, flattened_values), axis=0)
            i += windowSize
            j += windowSize
            iters += 1
        features = np.delete(features, 1, 0)
        return features

    @ staticmethod
    def gen_features_test(data):
        ftr_dim_size = 9*4  # number of features
        dim = len(data[0])
        # print("test:{}".format(len(data[i:j, 1])))
        mu = [np.mean(data[:, k]) for k in range(dim)]
        std = [np.std(data[:, k]) for k in range(dim)]
        var = [np.var(data[:, k]) for k in range(dim)]
        maxi = [max(data[:, k]) for k in range(dim)]
        mini = [min(data[:, k]) for k in range(dim)]
        min_max_diff = [maxi[k] - mini[k] for k in range(dim)]
        median = [np.median(data[:, k]) for k in range(dim)]
        count_above_mean = [np.count_nonzero(
            data[:, k] > mu[k]) for k in range(dim)]
        count_below_mean = [np.count_nonzero(
            data[:, k] < mu[k]) for k in range(dim)]
        values = np.array([
            mu,
            std,
            var,
            maxi,
            mini,
            min_max_diff,
            median,
            count_above_mean,
            count_below_mean
        ])
        return values.reshape(1, ftr_dim_size)

    def print_statistics(self):
        print(
            "Size of training set : {}, size of testing set: {}".format(
                len(self.xTr), len(self.xTe)))
        print("\nTraining split:\n Count of +1 labels: {}, Count of -1 labels: {}".format(
            np.count_nonzero(self.yTr + 1), np.count_nonzero(self.yTr - 1)))
        print("\nTesting split:\n Count of +1 labels: {}, Count of -1 labels: {}".format(
            np.count_nonzero(self.yTe + 1), np.count_nonzero(self.yTe - 1)))
