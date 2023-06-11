
"""
In this project, we implemented a class SVM (Support Vector Machine) to classify data with 4 different labels
using the OvO (One-vs-One) method. The SVM is a powerful machien learning algorithm, and the OvO method is a 
technique for multiclass classification that involves training muiltiple binary classifiers to distinguish
between pairs of classes.

We used cross-validation to find the best regularization parameter (lambda) for the SVM. Once we have found the 
best value of lambda, we will retrain our SVM using the OvO method using the best lambda, and then use this model
to make prediction.

"""

import numpy as np

class SVM:
    def __init__(self, learningRate=0.1, maxIteration=200, lambdaParam=0.5, nFolds=5):
        self.w = None # weights
        self.b = None # bias
        self.lr = learningRate 
        self.maxIter = maxIteration # max iteration times
        self.lambdaParam = lambdaParam 
        self.nFolds = nFolds

    def reset(self):
        """
        Reset paramaters (the weights and bias)in SVM classifier
        """

        self.w = None
        self.b = None

    def getInitParam(self, data, target):
        """
        Get initial paramters of the model
        """

        data = np.array(data)  # Convert data to an array

        # .sameples -> number of Data points -> number of rows
        # .features -> number of input features -> number of columns
        self.samples, self.features = data.shape

        # initiating the weight value and bias value
        self.w = np.zeros(self.features) 
        self.b = 0

        self.X = data
        self.y = target

        self.samples, self.features = data.shape

    def updateWeightAndBias(self, data, target):

        """
        Update weights and bias

        ---------------------------------------------------
    
        w_2 = w_1 - lr * dw
        b_2 = b_1 - lr * dw
    
        We use hidge loss function, which is defined as:
            J(y, f(x)) = max(0, 1 - y*f(x) + lambda*|w|^2)
            f(x) = w.T*x + b
    
    
        If 1 - y_i * (w.T*x_i - b) + lambda*|w|^2 <= 0:
            dw = 2 * labmda * w
            db = 0
    
        If 1 - y_i * (w.T*x_i - b) + lambda*|w|^2 > 0:
            dw = -y_i * x_i + 2 * labmda * w
            db = y_i
    
        """
        for idx, x_i in enumerate(data):
            condition = target[idx] * (np.dot(x_i, self.w) - self.b) >= 1

            # Compute the partial derivatives dw and db
            if condition:
                dw = 2 * self.lambdaParam * self.w
                db = 0
            else:
                dw = 2 * self.lambdaParam * self.w - np.dot(x_i, target[idx])
                db = target[idx]

            # Update the parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def svmFit(self, data, target):
        """
        Fit a linear SVM using gradient descent
        """
        # Covert X to an array
        data = np.array(data)
        
        # Get the initial parameter values
        self.getInitParam(data,target)

        # gradient descent
        for iter in range(self.maxIter):
            self.updateWeightAndBias(data, target)

    def svmPredict(self, data, legal=None):
        """
        Predict the class labels for a set of samples using a weight vector and bias
        """

        # Compute the predict value using the formula y_pred = w.T * X - b
        yPred = np.dot(data, self.w) - self.b
        return np.sign(yPred)

    def svmCrossValidation(self, data, target, lambdaList):
        """
        Use cross-validation to find the best value of lambda
        """

        foldSize = self.samples // self.nFolds # Compute the sample size for each fold
        theBestLambda = None
        theBestAcc = -1 # Set the lower bound of the best accuracy, here we set it to -1.

        for theLambda in lambdaList:
            acc = 0 

            for i in range(self.nFolds):
                # Split the data into training and validation set
                X_train = np.concatenate( [data[:i*foldSize, :], data[(i+1)*foldSize:, :]], axis=0 ) 
                y_train = np.concatenate( [target[:i*foldSize], target[(i+1)*foldSize:]], axis=0 )
                X_val = data[i*foldSize : (i+1)*foldSize, :]
                y_val = target[i*foldSize : (i+1)*foldSize]

                # Fit the SVM on the training data. Then compute the validation accuracy value 
                self.svmFit(X_train, y_train)
                y_pred = self.svmPredict(X_val)
                acc += np.mean(y_pred == y_val)
            
            acc = acc / self.nFolds # the average accuracy using the lambda

            if acc > theBestAcc:
                theBestAcc = acc
                theBestLambda = theLambda

        # Update the parameter labmda
        self.lambdaParam = theBestLambda

        # Fit the SVM model using the updated lambda
        self.svmFit(data, target)

            
    
    def fit(self, data, target):
        """
        Fit the SVM to the data
        """

        self.getInitParam(data, target)

        lambdaList = [0.01, 0.05, 0.1 ,0.5, 1, 5, 10, 50, 100] 

        # Find the best lambda using cross validation
        self.svmCrossValidation(data, target, lambdaList)

    def predict(self, data):
        """
        Predict the class labels for a set of samples
        """
        return self.svmPredict(data)


# One vs one SVM classifer
class Classifier:
    def __init__(self):
        self.classifiers = {}

    def reset(self):
        # Reset paramaters in OVO SVM classifier
        self.classifiers = {}

    def fit(self, data, target):
        """
        Train a One-vs-One SVM classifier
        ----------------------------------------------------------------------------------------------
        If nClasses = N, then N*(N-1)/2 binary classifiers are trained.
        Each classifier is trained on a subset of the data that contains samples from only 2 classes,
        and it learns to distinguish bewteen these 2 classes.
        """
        print("\nStart Training...")

        classes = np.unique(target) 
        nClasses = len(classes)

        # Training classifiers on a subset of data containing only 2 labels
        for label1 in range(nClasses - 1):
            for label2 in range(label1 + 1, nClasses):
                subData = np.array(data)[(target == classes[label1]) | (target == classes[label2])]
                subTarget = np.array(target)[(target == classes[label1]) | (target == classes[label2])]
                subTarget = np.where(subTarget == classes[label1], -1, 1)

                svm = SVM()
                svm.fit(subData, subTarget)
                self.classifiers[(classes[label1], classes[label2])] = svm
        
        print("\nTraining done.")

    def predict(self, data, legal=None):
        """
        Predict class labels for samples in data.
        -----------------------------------------------------------------------------------------------
        Each binary classifiers predicts the class label of a new data point.
        The class with the most votes is chosen as the final prediction for the data point
        """

        nClassifiers = len(self.classifiers)

        winnersArray = np.zeros(nClassifiers) # Array that records the winners of each classifier

        classifierIndex = 0 

        for label1, label2 in self.classifiers:
            svm = self.classifiers[(label1, label2)]
            winner = np.where(svm.predict(data) == -1, label1, label2)
            winnersArray[classifierIndex] = winner
            classifierIndex += 1

        theFinalPrediction = np.bincount(winnersArray.astype(int)).argmax()
        

        return theFinalPrediction
        
