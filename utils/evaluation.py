import torch
import numpy as np

def evalPCKh(predictions, target, threshold, alpha=0.5):

    predictions = 4*predictions.reshape(-1,2)
    target = 4*target.reshape(-1,2)
    assert predictions.shape == target.shape
    jointErr = np.sqrt(np.sum(np.square(predictions - target),1))
    meanErr = np.mean(jointErr)
    headSize = np.array(threshold*alpha)
    correctKeypoints = (headSize - jointErr).reshape(-1,1)
    correctKeypoints[correctKeypoints>=0] = 1
    correctKeypoints[correctKeypoints<0] = 0
    numJoints = jointErr.shape[0]
    numCorrect = np.sum(correctKeypoints)
    pckhResult = float(numCorrect/numJoints)
    return pckhResult, meanErr