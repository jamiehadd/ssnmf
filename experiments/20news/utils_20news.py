import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
plt.ion()

def top_features(classifier, feature_names, cls_names, top_num):
    """
    Print the classifier features.
    Args:
        classifier (object): a fitted classifier
        feature_names (list): vocabulary
        cls_names (list): class labels (str)
    """
    for i, cls in enumerate(cls_names):
        print("%s: %s" % (cls, ", ".join(np.asarray(feature_names)[np.argsort(classifier.coef_[i])[:-top_num-1:-1]])))

def print_keywords(W, features, top_num):
    """
    Print the keyword representation of each topic.
    Args:
        W (ndarray): matrix of the document representation of topics, shape (topics, documents)
        features (list): vocabulary
        n (int): number of top keywords to print
    """
    for idx, topic in enumerate(W):
        keywords = "Topic %d: " % (idx+1)
        keywords += ", ".join([features[i] for i in topic.argsort()[:-top_num-1:-1]])
        print(keywords)


def factors_heatmaps(B, cls_names = [], save=False, filepath = None):
    """
    Plot heatmap of the classifier coefficient matrix, shape (classes, topics).
    Args:
        B (ndarray): classifier coefficient matrix (classes, topics)
        cls_names (list): class labels (str)
        save (boolean): True to save figure, False otherwise
        filepath (str): save figure in filepath.
    """
    fig,ax = plt.subplots(figsize=(8,2))
    sns.heatmap(B)
    plt.xlabel('Topic', fontsize=14)
    plt.ylabel('Class', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation= 0)

    ax.set_yticklabels(cls_names)
    labels_x = [i+1 for i in range(B.shape[1])]
    ax.set_xticklabels(labels_x)

    if save == True:
        plt.savefig(filepath, bbox_inches='tight')
    #plt.show()

def TopicVsDoc(S):
    """
    This function associates a topic to each document (one-hot encode).
    Args:
         S (ndarray): document representation matrix, shape (topics, documents)
    Returns:
        topics_doc (ndarray): binary matrix where 1 in the (i,j) entry indicates that document j is associated the most with topic i
    """
    topics_doc = np.zeros(S.shape)
    for i in range(S.shape[1]):
        if not np.all((S[:,i] == 0)):
            max_index = np.argmax(S[:, i])
            topics_doc[max_index,i] = 1

    return topics_doc

def regression_metrics(true_target, pred_target):
        '''
        This function evaluates the regression model with various metrics.
        Args:
            true_target(ndarray): true targets, shape (1, n_samples) or (n_samples, 1) or (n_samples,)
            pred_target(ndarray): predicted targets, shape (1, n_samples) or (n_samples, 1) or (n_samples,)

        Returns:
            r2_val(float): R-squared value of regeression model
            mse_val(float): Mean Squared Error of regeression model
            mae_val(float): Mean Absolute Error of regeression model

        '''
        true_target = np.squeeze(true_target)
        pred_target = np.squeeze(pred_target)

        r2_val = r2_score(true_target, pred_target)
        mse_val = mean_squared_error(true_target, pred_target)
        mae_val = mean_absolute_error(true_target, pred_target)

        print("\nR-sq value is {:.5f}.".format(r2_val))
        print("MSE is {:.5f}.".format(mse_val))
        print("MAE is {:.5f}.\n".format(mae_val))

        return r2_val, mse_val, mae_val

def clust_scores(M, S, clust):
    """
    This function calculates clusetring scores.
    Args:
        M (ndarray): ground-truth representation matrix, shape (number of subgroups, number of documents)
        S (ndarray): SSNMF/NMF document representation matrix, shape (number of topics, number of documents)
        clust (str): "hard" for hard thresholding; "soft" for soft clustering
    Returns:
        score_max (list): clustering score (float) for each topic (i.e. row of S)
         I (list): the best subcategory index (i.e. row of M) (int) for each topic
    """
    if clust == "hard":
        S_mod = TopicVsDoc(S)
    if clust == "soft":
        col_sum = S.sum(axis=0)[None,:]
        col_sum[col_sum==0] = 1
        S_mod = S/col_sum

    score_max = [0] * S_mod.shape[0]
    I = [0] * S_mod.shape[0]

    for l in range(S_mod.shape[0]):
        S_l = S_mod[l,:]
        for i in range(M.shape[0]):
            M_i = M[i,:]
            score = np.sum(np.multiply(S_l,M_i))/np.sum(M_i)
            if (score > score_max[l]):
                score_max[l] = score
                I[l] = i
    return score_max, I

def clustering_analysis(subcat_onehot, sub_names, S_dict, S_test_dict, median_dict, clust, numb_train):
    """
    This function computes clusetring scores summaries.
    Args:
        subcat_onehot (ndarray): one-hot encoded ground-thruth subcategories labels, shape (subcategories, documents)
        sub_names (list): names (str) corresnoding to subcategories labels
        S_dict (dictionary): document dictionary matrices for each model for each iteration, shape (topics, train + test documents)
        S_test_dict (dictionary): document dictionary matrices for each model for each iteration, shape (topics, test documents)
        median_dict (dictionary): indices of the median model for each model based on test accuracy
        clust (str): "hard" for hard thresholding; "soft" for soft clustering
        numb_train (int): number of documents in train set

    """
    #SSNMF Models
    print("\n\nSSNMF Results:")
    for i in range(3,7):
        score_avg = []
        S_list = S_dict["Model" + str(i)]
        S_test_list = S_test_dict["Model" + str(i)]
        for i_avg in range(len(S_list)):
            S_train = S_list[i_avg][:,0:numb_train]
            S_test = S_test_list[i_avg]
            S_model = np.concatenate((S_train, S_test), axis = 1) # concatenate the representation for train and test data
            score_max, I = clust_scores(M = subcat_onehot, S = S_model, clust = clust)
            score_avg.append(np.mean(score_max))
        print("Average SSNMF Model {} score {}.".format(i,np.mean(score_avg)))

        # SSNMF Median Results
        S = S_dict["Model" + str(i)][median_dict["Model" + str(i)]]
        S_test = S_test_dict["Model" + str(i)][median_dict["Model" + str(i)]]
        S_train = S[:,0:numb_train]
        S_model = np.concatenate((S_train, S_test), axis = 1) # concatenate the representation for train and test data
        score_max, I = clust_scores(M = subcat_onehot, S = S_model, clust = clust)
        print("SSNMF Median Model {} Results:".format(i))
        for i_m in range(len(score_max)):
            print("Topic {}: {} (score: {})".format(i_m+1,sub_names[I[i_m]],score_max[i_m]))
        print("Median SSNMF Model {} average score {}.\n\n".format(i,np.mean(score_max)))

    # NMF models
    for nmf_model in ["NMF", "I_NMF"]:
        print("\n\n" + nmf_model + " Results:")
        H_train_list = S_dict[nmf_model]
        H_test_list = S_test_dict[nmf_model]
        score_nmf_avg = []
        for i_avg in range(len(H_train_list)):
            H_train = H_train_list[i_avg]
            H_test = H_test_list[i_avg]
            H_model = np.concatenate((H_train, H_test), axis = 1)
            score_max_nmf, I_nmf = clust_scores(M=subcat_onehot, S=H_model, clust = clust)
            score_nmf_avg.append(np.mean(score_max_nmf))
        print("Average " +nmf_model+ " Model score {}.".format(np.mean(score_nmf_avg)))

        # NMF Median Results
        H_train = S_dict[nmf_model][median_dict[nmf_model]]
        H_test = S_test_dict[nmf_model][median_dict[nmf_model]]
        H_model = np.concatenate((H_train, H_test), axis = 1)
        score_max_nmf, I_nmf = clust_scores(M=subcat_onehot, S=H_model, clust = clust)
        for i_m in range(len(score_max_nmf)):
            print("Topic {}: {} (score: {})".format(i_m+1,sub_names[I_nmf[i_m]],score_max_nmf[i_m]))
        print("Median " + nmf_model + " model average score {}.".format(np.mean(score_max_nmf)))

def dictupdateIdiv(Z, D, R, M, eps):
    '''
    multiplicitive update for D and R in D(Z||DR)
    Parameters
    ----------
    Z   : array
          Data matrix.
    D   : array
          Left factor matrix of Z.
    R   : array
          Right factor matrix of Z.
    M   : array
          Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
    eps : float_, optional
        Epsilon value to prevent division by zero (default is 1e-10).
    Returns
    -------
    updated D or the transpose of updated R
    '''
    return np.multiply(np.divide(D, eps + M @ np.transpose(R)), \
                       np.multiply(np.divide(np.multiply(M, Z), eps + np.multiply(M, D @ R)), M) @ np.transpose(R))
