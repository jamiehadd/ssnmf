import numpy as np
import SSNMF_torch
import torch
from scipy.optimize import nnls


class SSNMF_Reuters_t:

    def __init__(self, train_features, train_labels, test_features, test_labels, k, modelNum, **kwargs):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.modelNum = modelNum
        self.k = k

        if train_features.size(1) != train_labels.size(1):
            raise Exception('The column dimensions of train_features and train_labels are not equal.')

        self.rows = train_features.size(0)  # number of features
        self.cols = train_labels.size(1)  # number of samples in the train and test set
        self.classes = self.train_labels.size(0)  # number of classes in the data

        self.lam = kwargs.get('lam', 1)
        self.numiters = kwargs.get('numiters', 10)
        self.tol = kwargs.get('tol', 1e-4)
        self.iter_s = kwargs.get('iter_s', 20)
        self.cuda = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

        # initialize factor A
        self.A = kwargs.get('A', torch.rand(self.rows, k, dtype=torch.float64, device=self.cuda))
        # initialize factor S
        self.S = kwargs.get('S', torch.rand(k, self.cols, dtype=torch.float64, device=self.cuda))
        # initialize factor B
        self.B = kwargs.get('B', torch.rand(self.classes, k, dtype=torch.float64, device=self.cuda))

        if self.rows != self.A.size(0):
            raise Exception('The row dimension of A is not equal to the number of features in the data.')
        if self.k != self.A.size(1):
            raise Exception('The column dimension of A is not equal to k.')
        if self.k != self.S.size(0):
            raise Exception('The row dimension of S is not equal to k.')
        if self.cols != self.S.size(1):
            raise Exception(
                'The column dimension of S is not equal to the number of samples in the train and test data.')
        if self.B.size(0) != self.classes:
            raise Exception('The row dimension B is not equal not equal to the number of classes.')
        if self.B.size(1) != self.k:
            raise Exception('The column dimension of B is not k.')

        self.W_train = kwargs.get('W_train', torch.ones([self.rows, self.cols], dtype=torch.float64, device=self.cuda))
        if self.W_train.size(0) != self.rows:
            raise Exception('The row dimension W_train is not equal to the number of features in the data.')
        if self.W_train.size(1) != self.cols:
            raise Exception(
                'The column dimension of W_train is not equal to the number of samples in the train and test data.')

        self.W_test = kwargs.get('W_test', torch.ones([self.test_features.size(0), self.test_features.size(1)],
                                                      dtype=torch.float64, device=self.cuda))
        if self.W_test.size(0) != self.rows:
            raise Exception('The row dimension W_test is not equal to the number of features in the data.')
        if self.W_test.size(1) != self.test_features.size(1):
            raise Exception('The column dimension of W_test is not equal to the number of samples in the test data.')

        self.L = kwargs.get('L', torch.ones([self.classes, self.cols], dtype=torch.float64, device=self.cuda))
        if self.L.size(0) != self.classes:
            raise Exception('The row dimension of L is not equal to the number of classes in the data.')
        if self.L.size(1) != self.cols:
            raise Exception(
                'The column dimension of L is not equal to the number of samples in the train and test data.')

        self.model = SSNMF_torch.SSNMF_torch(X=self.train_features, k=self.k, modelNum=self.modelNum,
                                 Y=self.train_labels, W=self.W_train, L=self.L, A=self.A,
                                 B=self.B, S=self.S, lam=self.lam, tol=self.tol)

    def eval(self):
        '''
        This function fits the (S)SNMF model to the train data, and evaluates the performance of the model on the train and test data.
        Returns:
            train_evals (list): [ndarray(total reconstruction error (model objective function) on train data for each iteration),
                                 ndarray(data reconstruction error on train data for each iteration),
                                 ndarray(label reconstruction error on train data for each iteration),
                                 ndarray(classification accuracy on train data for each iteration)]
            test_evals (list): [float(total reconstruction error (model objective function) on test data),
                                float(data reconstruction error on test data),
                                float(label reconstruction error on test data),
                                float(classification accuracy on test data)]
        '''
        # Fit (S)SNMF model to train data
        train_evals = self.model.mult(numiters=self.numiters, saveerrs=False)  # save train data errors
        # Apply (S)SNMF model to test data
        self.S_test = self.test_model()  # representation matrix of test data
        # Compute reconstruction errors on test data
        test_evals = self.computeErrors()

        return test_evals

    def test_model(self):
        '''
        Given a trained (S)SNMF model i.e. learned data dictionary, A, the function applies the (S)SNMF model
        to the test data to produce the representation of the test data.
        Returns:
            S_test (ndarray): representation matrix of the test data, shape(#topics, #test features)
        '''
        if self.modelNum == 3 or self.modelNum == 4:  # Frobenius discrepancy measure on features data (use nonnegative least squares)
            S_test = torch.rand(self.k, self.test_features.size(1), dtype=torch.float64, device=self.cuda)
            for i in range(self.iter_s):
                S_test = torch.t(self.model.dictupdateFro(torch.t(self.test_features), torch.t(S_test),
                                                          torch.t(self.model.A), torch.t(self.W_test), eps=1e-10))

        if self.modelNum == 5 or self.modelNum == 6:  # I-divergence discrepancy measure on features data (use mult. upd. I-div)
            S_test = torch.rand(self.k, self.test_features.size(1), dtype=torch.float64, device=self.cuda)
            for i in range(self.iter_s):
                S_test = torch.t(self.model.dictupdateIdiv(torch.t(self.test_features), torch.t(S_test),
                                                           torch.t(self.model.A), torch.t(self.W_test), eps=1e-10))
        return S_test

    def computeErrors(self):

        test_labels_approx = self.model.B @ self.S_test
        test_labels_true = self.get_label_list()


        oneError = self.oneErr(test_labels_approx, test_labels_true)
        coverage = self.coverage(test_labels_approx, test_labels_true)
        averagePrecision = self.average_precision(test_labels_approx, test_labels_true)

        return [oneError, coverage, averagePrecision]

    def oneErr(self, test_labels_approx, test_labels_true):

        count = 0
        n = test_labels_approx.size(1)

        for i in range(n):

            labels_pre = test_labels_approx[:, i]
            labels_true = test_labels_true[i]

            if torch.argmax(labels_pre) not in labels_true:
                count += 1

        one_error = count/n

        return one_error

    def coverage(self, test_labels_approx, test_labels_true):

        coverage = 0
        n = test_labels_approx.size(1)

        for i in range(n):
            labels_true = test_labels_true[i]
            sorted_labels_indices = torch.argsort(test_labels_approx[:, i], dim=0, descending=True)
            max_cover = 0
            for j in range(len(labels_true)):
                max_cover = max(max_cover, (sorted_labels_indices == labels_true[j]).nonzero().item())
            coverage = coverage + max_cover

        return coverage/n

    def average_precision(self, test_labels_approx, test_labels_true):

        ave_precision = 0
        n = test_labels_approx.shape[1]

        for i in range(n):
            labels_true = test_labels_true[i]
            sorted_labels_indices = torch.argsort(test_labels_approx[:, i], dim=0, descending=True)

            if len(labels_true) == 1:
                ave_precision = ave_precision + 1 / ((sorted_labels_indices == labels_true[0]).nonzero().item() + 1)
            else:
                sorted_labels_rank = self.sorting_label_rank(labels_true, sorted_labels_indices)
                n_Y = len(labels_true)
                precision = 0
                for j in range(n_Y):
                    precision = precision + (j + 1) / (sorted_labels_rank[j] + 1)

                ave_precision = ave_precision + precision / n_Y

        return ave_precision/n

    def get_label_list(self):

        lists_of_labels = []
        for i in range(self.test_labels.size(1)):
            index_tuple = torch.where(self.test_labels[:, i] == 1)
            index_list = index_tuple[0].tolist()
            lists_of_labels.append(index_list)

        return lists_of_labels

    def sorting_label_rank(self, labels_true, sorted_labels_indices):
        labels_rank = []
        for i in range(len(labels_true)):
            labels_rank.append((sorted_labels_indices == labels_true[i]).nonzero().item())

        labels_rank.sort()
        return labels_rank

    def get_lables_prediction(self, test_labels_approx, threshold):
        for i in range(test_labels_approx.size(1)):
            r = max(test_labels_approx[:, i]) - min(test_labels_approx[:, i])
            test_labels_approx[:, i] = test_labels_approx[:, i] >= (min(test_labels_approx[:, i]) + threshold * r)
        return test_labels_approx












