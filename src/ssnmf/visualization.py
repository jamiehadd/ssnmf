# Import necessary packages
import numpy as np
import torch
#import torchvision
import matplotlib.pyplot as plt
from time import time
import os
#from google.colab import drive
import scipy.optimize.nnls as nnls
from numpy import linalg as la

import ssnmf

def topic_plot(A, vertpixels, horizpixels, colnum):
    topic = np.transpose(np.reshape(A[:,colnum],[horizpixels,vertpixels]))
    plt.imshow(topic, cmap='binary')
    plt.show()

def visualize_reconstr(dictionary, representation, vertpixels, horizpixels, indices):
    # Indices is a list
    recon = np.matmul(dictionary,representation)
    for index in indices:
        image = np.transpose(np.reshape(recon[:,index],[horizpixels,vertpixels]))
        plt.imshow(image, cmap='binary')
        plt.savefig('./'+'reconstruction'+str(index)+'.png')



def plot_util(kchoices,train_errs,test_errs,train_reconerrs,test_reconerrs,train_classerrs,test_classerrs,train_accs,test_accs,namme,iOption):
    # iOption - indicate which x-axis label to use - options are 'k','l','n'
    # namme   - Name of figure to be saved. 

    fig, axs = plt.subplots(2, 2, figsize=(17,10))
    axs[0, 0].plot(kchoices,train_errs,color='blue',linewidth=4,label='Train Errors')
    axs[0, 0].plot(kchoices,test_errs,color='red',linestyle='dashed',linewidth=4,label='Test Errors')
    axs[0, 0].legend()
    axs[0, 0].set_title('Train Errors')

    axs[0, 1].plot(kchoices,train_reconerrs,color='blue',linewidth=4,label='Train Reconstruction Errors')
    axs[0, 1].plot(kchoices,test_reconerrs,color='red',linestyle='dashed',linewidth=4,label='Test Reconstruction Errors')
    axs[0, 1].legend()
    axs[0, 1].set_title('Train Reconstruction Errors')

    axs[1, 0].plot(kchoices,train_classerrs,color='blue',linewidth=4,label='Train Classification Errors')
    axs[1, 0].plot(kchoices,test_classerrs,color='red',linestyle='dashed',linewidth=4,label='Test Classification Errors')
    axs[1, 0].legend()
    axs[1, 0].set_title('Train Classification Errors')
    
    axs[1, 1].plot(kchoices,train_accs,color='blue',linewidth=4,label='Train Classification Accuracies')
    axs[1, 1].plot(kchoices,test_accs,color='red',linestyle='dashed',linewidth=4,label='Test Classification Accuracies')
    axs[1, 1].legend()
    axs[1, 1].set_title('Train Classification Accuracies')

    if(iOption == 'k'):
        axs[0, 0].set_xlabel('k')
        axs[0, 1].set_xlabel('k')
        axs[1, 0].set_xlabel('k')
        axs[1, 1].set_xlabel('k')

    if(iOption == 'l'):
        axs[0, 0].set_xlabel('lambda')
        axs[0, 1].set_xlabel('lambda')
        axs[1, 0].set_xlabel('lambda')
        axs[1, 1].set_xlabel('lambda')

    if(iOption == 'n'):
        axs[0, 0].set_xlabel('Iterations')
        axs[0, 1].set_xlabel('Iterations')
        axs[1, 0].set_xlabel('Iterations')
        axs[1, 1].set_xlabel('Iterations')

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper center')
    plt.savefig('./'+namme+'.png')


def k_plots(train_features, train_labels, test_features, test_labels, kchoices, lam, numiters,avgnum):
    numks = np.shape(kchoices)[0]
    train_errs = [0]*numks
    train_reconerrs = [0]*numks
    train_classerrs = [0]*numks
    train_accs = [0]*numks
    test_errs = [0]*numks
    test_reconerrs = [0]*numks
    test_classerrs = [0]*numks
    test_accs = [0]*numks

    for i in range(numks):
        for j in range(avgnum):
            module = TrainTestSetEvaluation(train_features, train_labels, test_features, test_labels, kchoices[i], lam, numiters)
            train_model_error, train_acc, numiters, [test_err, test_reconerr, test_classerr, test_acc], S_test = module.tt_eval_ssnmfmult()
            train_model_errors = train_model_error[0]
            train_errs[i] = train_errs[i]+train_model_errors[numiters-1]
            train_model_reconerrs = train_model_error[1]
            train_reconerrs[i] = train_reconerrs[i]+train_model_reconerrs[numiters-1]
            train_model_classerrs = train_model_error[2]
            train_classerrs[i] = train_classerrs[i]+train_model_classerrs[numiters-1]
            train_accs[i] = train_accs[i]+train_acc
            test_errs[i] = test_errs[i]+test_err
            test_reconerrs[i] = test_reconerrs[i]+test_reconerr
            test_classerrs[i] = test_classerrs[i]+test_classerr
            test_accs[i] = test_accs[i]+test_acc

    train_errs = [element/avgnum for element in train_errs]
    train_reconerrs = [element/avgnum for element in train_reconerrs]
    train_classerrs = [element/avgnum for element in train_classerrs]
    train_accs = [element/avgnum for element in train_accs]
    test_errs = [element/avgnum for element in test_errs]
    test_reconerrs = [element/avgnum for element in test_reconerrs]
    test_classerrs = [element/avgnum for element in test_classerrs]
    test_accs = [element/avgnum for element in test_accs]

    plot_util(kchoices,train_errs,test_errs,train_reconerrs,test_reconerrs,train_classerrs,test_classerrs,train_accs,test_accs,'k_SSNMF','k')

def kl_k_plots(train_features, train_labels, test_features, test_labels, kchoices, lam, numiters,avgnum):
    numks = np.shape(kchoices)[0]
    train_errs = [0]*numks
    train_reconerrs = [0]*numks
    train_classerrs = [0]*numks
    train_accs = [0]*numks
    test_errs = [0]*numks
    test_reconerrs = [0]*numks
    test_classerrs = [0]*numks
    test_accs = [0]*numks

    for i in range(numks):
        for j in range(avgnum):
            module = TrainTestSetEvaluation(train_features, train_labels, test_features, test_labels, kchoices[i], lam, numiters)
            train_model_error, train_acc, numiters, [test_err, test_reconerr, test_classerr, test_acc], S_test = module.tt_eval_kl_ssnmfmult()
            train_model_errors = train_model_error[0]
            train_errs[i] = train_errs[i]+train_model_errors[numiters-1]
            train_model_reconerrs = train_model_error[1]
            train_reconerrs[i] = train_reconerrs[i]+train_model_reconerrs[numiters-1]
            train_model_classerrs = train_model_error[2]
            train_classerrs[i] = train_classerrs[i]+train_model_classerrs[numiters-1]
            train_accs[i] = train_accs[i]+train_acc
            test_errs[i] = test_errs[i]+test_err
            test_reconerrs[i] = test_reconerrs[i]+test_reconerr
            test_classerrs[i] = test_classerrs[i]+test_classerr
            test_accs[i] = test_accs[i]+test_acc

    train_errs = [element/avgnum for element in train_errs]
    train_reconerrs = [element/avgnum for element in train_reconerrs]
    train_classerrs = [element/avgnum for element in train_classerrs]
    train_accs = [element/avgnum for element in train_accs]
    test_errs = [element/avgnum for element in test_errs]
    test_reconerrs = [element/avgnum for element in test_reconerrs]
    test_classerrs = [element/avgnum for element in test_classerrs]
    test_accs = [element/avgnum for element in test_accs]

    plot_util(kchoices,train_errs,test_errs,train_reconerrs,test_reconerrs,train_classerrs,test_classerrs,train_accs,test_accs,'k_KLSSNMF','k')
    
def lam_plots(train_features, train_labels, test_features, test_labels, k, lamchoices, numiters,avgnum):
    numlams = np.shape(lamchoices)[0]
    train_errs = [0]*numlams
    train_reconerrs = [0]*numlams
    train_classerrs = [0]*numlams
    train_accs = [0]*numlams
    test_errs = [0]*numlams
    test_reconerrs = [0]*numlams
    test_classerrs = [0]*numlams
    test_accs = [0]*numlams

    for i in range(numlams):
        for j in range(avgnum):
            module = TrainTestSetEvaluation(train_features, train_labels, test_features, test_labels, k, lamchoices[i], numiters)
            train_model_error, train_acc, numiters, [test_err, test_reconerr, test_classerr, test_acc], S_test = module.tt_eval_ssnmfmult()
            train_model_errors = train_model_error[0]
            train_errs[i] = train_errs[i]+train_model_errors[numiters-1]
            train_model_reconerrs = train_model_error[1]
            train_reconerrs[i] = train_reconerrs[i]+train_model_reconerrs[numiters-1]
            train_model_classerrs = train_model_error[2]
            train_classerrs[i] = train_classerrs[i]+train_model_classerrs[numiters-1]
            train_accs[i] = train_accs[i]+train_acc
            test_errs[i] = test_errs[i]+test_err
            test_reconerrs[i] = test_reconerrs[i]+test_reconerr
            test_classerrs[i] = test_classerrs[i]+test_classerr
            test_accs[i] = test_accs[i]+test_acc

    train_errs = [element/avgnum for element in train_errs]
    train_reconerrs = [element/avgnum for element in train_reconerrs]
    train_classerrs = [element/avgnum for element in train_classerrs]
    train_accs = [element/avgnum for element in train_accs]
    test_errs = [element/avgnum for element in test_errs]
    test_reconerrs = [element/avgnum for element in test_reconerrs]
    test_classerrs = [element/avgnum for element in test_classerrs]
    test_accs = [element/avgnum for element in test_accs]

    plot_util(lamchoices,train_errs,test_errs,train_reconerrs,test_reconerrs,train_classerrs,test_classerrs,train_accs,test_accs,'l_SSNMF','l')

def kl_lam_plots(train_features, train_labels, test_features, test_labels, k, lamchoices, numiters,avgnum):
    numlams = np.shape(lamchoices)[0]
    train_errs = [0]*numlams
    train_reconerrs = [0]*numlams
    train_classerrs = [0]*numlams
    train_accs = [0]*numlams
    test_errs = [0]*numlams
    test_reconerrs = [0]*numlams
    test_classerrs = [0]*numlams
    test_accs = [0]*numlams

    for i in range(numlams):
        for j in range(avgnum):
            module = TrainTestSetEvaluation(train_features, train_labels, test_features, test_labels, k, lamchoices[i], numiters)
            train_model_error, train_acc, numiters, [test_err, test_reconerr, test_classerr, test_acc], S_test = module.tt_eval_kl_ssnmfmult()
            train_model_errors = train_model_error[0]
            train_errs[i] = train_errs[i]+train_model_errors[numiters-1]
            train_model_reconerrs = train_model_error[1]
            train_reconerrs[i] = train_reconerrs[i]+train_model_reconerrs[numiters-1]
            train_model_classerrs = train_model_error[2]
            train_classerrs[i] = train_classerrs[i]+train_model_classerrs[numiters-1]
            train_accs[i] = train_accs[i]+train_acc
            test_errs[i] = test_errs[i]+test_err
            test_reconerrs[i] = test_reconerrs[i]+test_reconerr
            test_classerrs[i] = test_classerrs[i]+test_classerr
            test_accs[i] = test_accs[i]+test_acc

    train_errs = [element/avgnum for element in train_errs]
    train_reconerrs = [element/avgnum for element in train_reconerrs]
    train_classerrs = [element/avgnum for element in train_classerrs]
    train_accs = [element/avgnum for element in train_accs]
    test_errs = [element/avgnum for element in test_errs]
    test_reconerrs = [element/avgnum for element in test_reconerrs]
    test_classerrs = [element/avgnum for element in test_classerrs]
    test_accs = [element/avgnum for element in test_accs]

    plot_util(lamchoices,train_errs,test_errs,train_reconerrs,test_reconerrs,train_classerrs,test_classerrs,train_accs,test_accs,'l_KLSSNMF','l')
    
def numiters_plots(train_features, train_labels, test_features, test_labels, k, lam, numiterschoices,avgnum):
    numnums = np.shape(numiterschoices)[0]
    train_errs = [0]*numnums
    train_reconerrs = [0]*numnums
    train_classerrs = [0]*numnums
    train_accs = [0]*numnums
    test_errs = [0]*numnums
    test_reconerrs = [0]*numnums
    test_classerrs = [0]*numnums
    test_accs = [0]*numnums

    for i in range(numnums):
        for j in range(avgnum):
            module = TrainTestSetEvaluation(train_features, train_labels, test_features, test_labels, k, lam, numiterschoices[i])
            train_model_error, train_acc, numiters, [test_err, test_reconerr, test_classerr, test_acc], S_test = module.tt_eval_ssnmfmult()
            train_model_errors = train_model_error[0]
            train_errs[i] = train_errs[i]+train_model_errors[numiters-1]
            train_model_reconerrs = train_model_error[1]
            train_reconerrs[i] = train_reconerrs[i]+train_model_reconerrs[numiters-1]
            train_model_classerrs = train_model_error[2]
            train_classerrs[i] = train_classerrs[i]+train_model_classerrs[numiters-1]
            train_accs[i] = train_accs[i]+train_acc
            test_errs[i] = test_errs[i]+test_err
            test_reconerrs[i] = test_reconerrs[i]+test_reconerr
            test_classerrs[i] = test_classerrs[i]+test_classerr
            test_accs[i] = test_accs[i]+test_acc

    train_errs = [element/avgnum for element in train_errs]
    train_reconerrs = [element/avgnum for element in train_reconerrs]
    train_classerrs = [element/avgnum for element in train_classerrs]
    train_accs = [element/avgnum for element in train_accs]
    test_errs = [element/avgnum for element in test_errs]
    test_reconerrs = [element/avgnum for element in test_reconerrs]
    test_classerrs = [element/avgnum for element in test_classerrs]
    test_accs = [element/avgnum for element in test_accs]

    plot_util(numiterschoices,train_errs,test_errs,train_reconerrs,test_reconerrs,train_classerrs,test_classerrs,train_accs,test_accs,'n_SSNMF','n')

def kl_numiters_plots(train_features, train_labels, test_features, test_labels, k, lam, numiterschoices,avgnum):
    numnums = np.shape(numiterschoices)[0]
    train_errs = [0]*numnums
    train_reconerrs = [0]*numnums
    train_classerrs = [0]*numnums
    train_accs = [0]*numnums
    test_errs = [0]*numnums
    test_reconerrs = [0]*numnums
    test_classerrs = [0]*numnums
    test_accs = [0]*numnums

    for i in range(numnums):
        for j in range(avgnum):
            module = TrainTestSetEvaluation(train_features, train_labels, test_features, test_labels, k, lam, numiterschoices[i])
            train_model_error, train_acc, numiters, [test_err, test_reconerr, test_classerr, test_acc], S_test = module.tt_eval_kl_ssnmfmult()
            train_model_errors = train_model_error[0]
            train_errs[i] = train_errs[i]+train_model_errors[numiters-1]
            train_model_reconerrs = train_model_error[1]
            train_reconerrs[i] = train_reconerrs[i]+train_model_reconerrs[numiters-1]
            train_model_classerrs = train_model_error[2]
            train_classerrs[i] = train_classerrs[i]+train_model_classerrs[numiters-1]
            train_accs[i] = train_accs[i]+train_acc
            test_errs[i] = test_errs[i]+test_err
            test_reconerrs[i] = test_reconerrs[i]+test_reconerr
            test_classerrs[i] = test_classerrs[i]+test_classerr
            test_accs[i] = test_accs[i]+test_acc

    train_errs = [element/avgnum for element in train_errs]
    train_reconerrs = [element/avgnum for element in train_reconerrs]
    train_classerrs = [element/avgnum for element in train_classerrs]
    train_accs = [element/avgnum for element in train_accs]
    test_errs = [element/avgnum for element in test_errs]
    test_reconerrs = [element/avgnum for element in test_reconerrs]
    test_classerrs = [element/avgnum for element in test_classerrs]
    test_accs = [element/avgnum for element in test_accs]

    plot_util(numiterschoices,train_errs,test_errs,train_reconerrs,test_reconerrs,train_classerrs,test_classerrs,train_accs,test_accs,'n_KLSSNMF','n')
