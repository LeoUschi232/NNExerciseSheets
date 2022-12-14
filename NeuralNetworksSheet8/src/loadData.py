import scipy.io as sio
import numpy as np
import tempfile
import os
import urllib3
urllib3.disable_warnings()
import requests


def loadData(filePath, nTest=10000):
    '''
    Examples:
    `loadData('/WWW/users/nn/affNIST/training_and_validation-orig.mat')`
    `loadData('https://users.ph.tum.de/nn/affNIST/training_and_validation-orig.mat')`
    `loadData('https://users.ph.tum.de/nn/affNIST/training_and_validation_batches/1.mat')`
    
    @param: filePath Path to a local file or URL
    @param: nTest Number of test data images, the rest is used for the training data
    '''
    
    tmpFile = None
    if 'http://' in filePath or 'https://' in filePath:
        tmpFile = tempfile.NamedTemporaryFile(delete=False)
        r = requests.get(filePath, allow_redirects=True, verify=False)
        print("Downloading file...")
        tmpFile.write(r.content)
        print("Done")
        filePath = tmpFile.name
    data = sio.loadmat(filePath)["affNISTdata"]
    if tmpFile is not None:
        os.unlink(filePath)

    data = data[0,0]
    images = data[2].swapaxes(0,1)
    digitsvect = data[4].swapaxes(0,1)
    digitsint = data[5][0]
    imageDim = int(np.sqrt(images.shape[1]))
    imageShape = (imageDim, imageDim)
    images = images.reshape(images.shape[0], imageDim, imageDim)
    nClasses = digitsvect.shape[1]
    xTrain = images[:images.shape[0]-nTest]
    yTrain = digitsvect[:images.shape[0]-nTest]
    xTest = images[images.shape[0]-nTest:]
    yTest = digitsvect[images.shape[0]-nTest:]
    return imageShape, nClasses, xTrain, yTrain, xTest, yTest


def loadDataForNielsen(filePath, nTest=10000):
    '''
    See `loadData` for the description.
    '''
    
    imageShape, nClasses, xTrain, yTrain, xTest, yTest = loadData(filePath, nTest)
    
    training_data=[]
    for i in range(xTrain.shape[0]):
        training_data.append(((xTrain[i].flatten()/255).reshape(-1,1), yTrain[i].flatten().reshape(-1,1)))
    test_data=[]
    for i in range(xTest.shape[0]):
        test_data.append(((xTest[i].flatten()/255).reshape(-1,1), np.argmax(yTest[i].flatten())))
    images = np.concatenate([xTrain, xTest]).reshape(xTrain.shape[0]+xTest.shape[0], xTrain.shape[1]*xTrain.shape[2])
    return images, training_data, test_data