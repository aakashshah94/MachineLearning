from random import seed

from math import exp
from random import random
from random import randrange
from collections import Counter
from csv import reader
import sys
import math


def createNetwork(inputs, outputs, hidden, *args):
    network = list()
    i = inputs
    global hiddenlayers
    global count5
    hiddenlayers = []
    if len(args)<hidden:
        sys.exit(0)

    for arg in args:
        hiddenlayers.append(arg)

    for u in range(len(hiddenlayers)):
        hiddenLayer = [{'Weights':[round(random(),3) for i in range(inputs + 1)]} for i in range(hiddenlayers[u])]
        network.append(hiddenLayer)
        inputs = hiddenlayers[u]
    outputLayer = [{'Weights':[round(random(),3) for i in range(inputs + 1)]} for i in range(outputs)]
    network.append(outputLayer)
    count5 = sum(hiddenlayers) + 1
    count10 = len(hiddenlayers)
    return network

def readData(file):
    global flag
    dataSet = list()
    asd = list()
    with open(file, 'r') as file:
        csvReader = reader(file)
        for row in csvReader:
            if not row:
                continue
            dataSet.append(row)
            seen = set()  # set of seen values, which starts out empty

        for lst in row[-1]:
            deduped = [x for x in lst if x not in seen]  # filter out previously seen values
            seen.update(deduped)  # add the new values to the set
        call = len(seen)
        if (call == 1) or (call == 2) or (call == 3) or (call == 4):
            flag = 1
        else:
            flag = 0
    return dataSet

def strColumnToFloat(dataSet, column):
    for row in dataSet:
        row[column] = float(row[column].strip())

def strColumnToInt(dataSet, column):
    class_values = [row[column] for row in dataSet]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataSet:
        row[column] = lookup[row[column]]
    return lookup

def dataSetMinMax(dataSet):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataSet)]
    return stats

def normalizeDataSet(dataSet, minmax):
    for row in dataSet:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def crossValidation(dataSet, nFolds):
    dataSetSplit = list()
    dataSet1 = list(dataSet)
    foldSize = int(len(dataSet) / nFolds)
    for i in range(nFolds):
        fold = list()
        while len(fold) < foldSize:
            index = randrange(len(dataSet1))
            fold.append(dataSet1.pop(index))
        dataSetSplit.append(fold)
    # print ('Split ',dataSetSplit)
    return dataSetSplit


def calcAccuracy(actual, predicted):
    if(flag == 1):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
    else:
        correct = 0
        for i in range(len(actual)):
            if (actual[i] == predicted[i]) or ((actual[i] <= predicted[i] + 75) and (actual[i] >= predicted[i] - 75)):
                correct += 1
    return correct / float(len(actual)) * 100.0

def evalAlgorithm(dataSet, algorithm, nFolds, *args):
    folds = crossValidation(dataSet, nFolds)
    scores = list()
    scores1 = list()
    for fold in folds:
        train1 = list(folds)
        train1.remove(fold)
        train1 = sum(train1, [])
        test1 = list()
        for row in fold:
            row1 = list(row)
            test1.append(row1)
        predicted, predicted1 = algorithm(train1, test1, *args)
        actual = [row[-1] for row in fold]
        actual1 = [row[-1] for row in train1]
        accuracy = calcAccuracy(actual, predicted)
        accuracy1 = calcAccuracy(actual1, predicted1)
        # mse = MSE(actual,predicted)
        # print mse
        scores.append(accuracy)
        scores1.append(accuracy1)
    return scores, scores1

def bias(weights, inputs):
    activate = weights[-1]
    for i in range(len(weights)-1):
        activate += weights[i] * inputs[i]
        #print(activate)
    return activate

def sigmoidActivation(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def forwardPropagation(network, row):
    global inputs
    inputs = row
    for layer in network:
        # print(layer)
        newInputs = []
        for neuron in layer:
            activate = bias(neuron['Weights'], inputs)
            neuron['Output'] = sigmoidActivation(activate)
            newInputs.append(neuron['Output'])
        inputs = newInputs
    return inputs

def sigmoidActivationDerivative(output):
    return (output * (1 - output))

def backPropagationError(network, expected):
    global errors
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['Weights'][j]*neuron['Delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['Output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['Delta'] = errors[j] * sigmoidActivationDerivative(neuron['Output'])


def updateWeights(network, row, learningRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['Output'] for neuron in network[i-1]]
        for neuron in network[i]:
            # red = 0
            for j in range(len(inputs)):
                # red += 1
                neuron['Weights'][j] += learningRate * neuron['Delta'] * inputs[j]
            neuron['Weights'][-1] += learningRate * neuron['Delta']
        # print('Neuron %d weights :' % i ,neuron['Weights'])


def trainNetwork(network, trainingDataSet, learningRate, maxIterations, expectedOutputs):
    global foldcount
    foldcount=foldcount+1
    print('Training for fold %d / %d in Progress...' % (foldcount, nFolds1))
    for iterations in range(maxIterations):
        errorSum = 0
        for row in trainingDataSet:
            outputs = forwardPropagation(network, row)
            expected = [0 for i in range(expectedOutputs)]
            # print('ROW', row)
            # print('ExpectedRow', len(expected))
            if flag == 1:
                expected[row[-1]] = 1
            else:
                if(int(row[-1])<len(expected)):
                    expected[int(row[-1])] = 1
            errorSum += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            backPropagationError(network, expected)
            updateWeights(network, row, learningRate)
        # print(network)
        # print(network)
        print('Fold : %d / %d >Iteration: %d, Learning Rate: %0.2f, Error: %0.3f' % (foldcount,nFolds1,iterations, learningRate, errorSum))
        # print ('Expected: ', expected, 'Got: ', outputs)

def predictOutcome(network, row):
    outputs = forwardPropagation(network,row)
    prediction = outputs.index(max(outputs))
    return prediction

def backPropagation(train1, test1, learningRate, iterations, hidden):
    global count1
    global count2
    global count3
    global count4
    inputs = len(train1[0]) - 1
    outputs = len(set([row[-1] for row in train1]))
    # outputs = 1
    # network = createNetwork(inputs, outputs, hidden, hidden1, hidden2)
    if(count10==1):
        network = createNetwork(inputs, outputs, hidden, hidden1)
    elif(count10==2):
        network = createNetwork(inputs, outputs, hidden, hidden1, hidden2)
    elif (count10 == 3):
        network = createNetwork(inputs, outputs, hidden, hidden1, hidden2, hidden3)
    elif (count10 == 4):
        network = createNetwork(inputs, outputs, hidden, hidden1, hidden2, hidden3, hidden4)
    else:
        network = createNetwork(inputs, outputs, hidden, hidden1, hidden2, hidden3, hidden4, hidden5)

    trainNetwork(network, train1, learningRate, iterations, outputs)
    predictions1 = list()
    for row in train1:
        prediction1 = predictOutcome(network, row)
        predictions1.append(prediction1)
    #     print('Expected: %d, Got: %d' % (row[-1], prediction1))
    #     count3 += 1
    #     if(prediction1 == row[-1]):
    #         count4 += 1
    # acc = count4*100/count3
    # print (count3, count4)
    # print('Training Accuracy: ',acc)
    # print(predictions1)
    # print('Training Accuracy: %.3f%%'% (sum(predictions1) / float(len(predictions1))))
    predictions = list()
    for row in test1:
        prediction = predictOutcome(network, row)
        # if(flag == 1):
        #     print('Expected=%d, Got=%d' % (row[-1], prediction))
        # else:
        #     print('Expected=%.3f, Got=%.3f' % (row[-1]*10, prediction))
        # count1 += 1
        predictions.append(prediction)
    #     if (prediction == row[-1]):
    #         count2 += 1
    # acc1 = count2 * 100 / count1
    # print (count1, count2)
    # #print('Test Accuracy: %0.3f%%' % ((count2 * 100) / count1))
    # print('Test Accuracy: ',acc1)
    # count5 = inputs + 1 + sum(h)
    count6 = 0
    count7 = 0
    # count8 = 0
    for i in range(len(network)):
        count7 += 1
        count8 = 0
        for neuron in network[i]:
            count6 += 1
            count8 += 1
            if(count6<=count5):
                # print(count6,count5)
                if(count6==count5):
                    print('Output Layer: Neuron 1 Weights')
                    print(neuron['Weights'])
                else:
                    print('Hidden Layer: %d, Neuron %d Weights:' % (count7,count8))
                    print(neuron['Weights'])
    # print(network)
    # print (network[0][1])
    return(predictions, predictions1)
# global network
def main(args):
    seed(1)
    global dataSet

    global trainPer
    file = args[1] # Argument 1 : complete path of the post-processed input dataset
    dataSet = readData(file)
    global hidden1
    global hidden2
    global hidden3
    global hidden4
    global hidden5
    global count10
    global  foldcount
    foldcount=0

    # train1 = readData('train_adult.csv')
# test1 = readData('test_adult.csv')
    if(flag == 1 or flag == 0):
        for i in range(len(dataSet[0])-1):
            strColumnToFloat(dataSet,i)
        strColumnToInt(dataSet,len(dataSet[0])-1)
        global nFolds1
        nFolds1 = 0
        trainPer = int(args[2])  # Argument 2 : percentage of the dataset to be used for training
        global errorTol
        errorTol= float(args[3])  # Argument 3 : acceptable value of error i.e. the value of error metric at which the algorithm can be terminated
        count10 = int(args[4])  # Argument 4 : number of hidden layers

        nFolds1 = int(100/(100-trainPer))
        if (count10 == 1):
            hidden1 = int(args[5])
        elif (count10 == 2):
            hidden1 = int(args[5])
            hidden2 = int(args[6])
        elif (count10 == 3):
            hidden1 = int(args[5])
            hidden2 = int(args[6])
            hidden3 = int(args[7])
        elif (count10 == 4):
            hidden1 = int(args[5])
            hidden2 = int(args[6])
            hidden3 = int(args[7])
            hidden4 = int(args[8])
        else:
            hidden1 = int(args[5])
            hidden2 = int(args[6])
            hidden3 = int(args[7])
            hidden4 = int(args[8])
            hidden5 = int(args[9])

        learningRate = 0.3
        maxIterations = 100
        scores, scores1 = evalAlgorithm(dataSet, backPropagation, nFolds1, learningRate, maxIterations, count10)
        print('Training Accuracy: %.3f%%' % (sum(scores1) / float(len(scores1))))
        print('Test Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
main(sys.argv)