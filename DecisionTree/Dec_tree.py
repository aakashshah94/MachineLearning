import csv
import math
import random
import sys


class Node:
    def __init__(self, attSplitIndex=0, attSplitValue=2, possibleIndex=[], negNum=0, posNum=0, check=2,
                 attributeSplitIndex=0, falseChild=None, trueChild=None):
        self.attSplitIndex = attSplitIndex
        self.attSplitValue = attSplitValue
        self.possibleIndex = possibleIndex
        self.posNum = posNum
        self.negNum = negNum
        self.check = check
        self.attributeSplitIndex = attributeSplitIndex
        self.falseChild = falseChild
        self.trueChild = trueChild


accThresh = 0
nodeNum = 0
leafNum = 0


def readData(file):
    read = csv.reader(open(file))
    isTitle = True
    global rowNum
    global attValues
    global attNum
    global comp

    attValues = []
    attList = []
    attNum = 0
    rowNum = 0

    for row in read:
        if isTitle:
            attNum = len(row) - 1
            comp = row[0:attNum]
            isTitle = False
        else:
            rowNum += 1
            t1 = (row[0:len(row) - 1], row[len(row) - 1])
            attValues.append(t1)


def tree():
    global root
    root = Node()
    pos = 0
    neg = 0
    # print("Number %d" %rowNum)
    for x in range(rowNum):
        x1 = attValues[x][1][0]
        if x1 == '0':
            pos += 1
        else:
            neg += 1
    # print("root +: %d"%p)
    # print("root -: %d"%n)
    root.posNum = pos
    root.negNum = neg
    root.possibleIndex = range(attNum)
    root = createTree(root, attValues)


def createTree(node, attList):
    if node.check == 2:
        global nodeNum
        global leafNum
        nodeNum += 1
        isOk = False
        bestGain = 0
        possibleIndex = []
        for x in range(len(node.possibleIndex)):
            possibleIndex.append(node.possibleIndex[x])

        bestCount = [[0, 0], [0, 0]]
        iGain = 0
        Entropy = calculateEntropy(node.negNum, node.posNum)
        if (Entropy == 0):
            isOk = True
        if ((not node.possibleIndex) or isOk):
            leafNum += 1
            if (node.negNum > node.posNum):
                node.check = 0
            else:
                if (node.negNum < node.posNum):
                    node.check = 1
                else:
                    node.check = random.randrange(1)
                    # print('CLASSIFIED as :%d'%node.check)
            return node
        bestAttIndex = possibleIndex[0]
        best0AttList = []
        best1AttList = []

        # print("Node +: %d"%node.posNum)
        # print("Node -: %d"%node.negNum)
        # print("Node Entropy: %f"%Entropy)

        # print("Possible attribute index: "+str(possibleIndex))

        for attributeIndex in possibleIndex:
            normal0AttList = []
            normal1AttList = []
            count = [[0, 0], [0, 0]]
            for x in range(len(attList)):
                x1 = attList[x][0][attributeIndex]
                x2 = attList[x][1][0]
                if x1 == '0':
                    normal0AttList.append(attList[x])
                    if x2 == '0':
                        count[0][0] += 1
                    else:
                        count[0][1] += 1
                else:
                    normal1AttList.append(attList[x])
                    if x2 == '0':
                        count[1][0] += 1
                    else:
                        count[1][1] += 1
            # Information Gain
            # print('Count status: '+str(count))
            normal0Entropy = calculateEntropy(count[0][0], count[0][1])
            normal1Entropy = calculateEntropy(count[1][0], count[1][1])
            normal0Weight = (count[0][0] + count[0][1]) / float(count[0][0] + count[0][1] + count[1][0] + count[1][1])
            normal1Weight = (count[1][0] + count[1][1]) / float(count[0][0] + count[0][1] + count[1][0] + count[1][1])
            entropy = normal0Entropy * normal0Weight + normal1Entropy * normal1Weight
            iGain = Entropy - entropy
            # print('Entropy with -'+str(attributeIndex)+ ': '+str(entropy))

            if (iGain > bestGain):
                bestGain = iGain
                bestAttIndex = attributeIndex
                bestCount = count
                best0AttList = normal0AttList
                best1AttList = normal1AttList

        # print("Best:")
        # print(bestGain)
        # print(bestAttIndex)
        # print(bestCount)
        # print("Possible"+str(possibleIndex))

        possibleIndex.remove(bestAttIndex)
        node.attributeSplitIndex = bestAttIndex
        node.falseChild = createTree(Node(bestAttIndex, 0, possibleIndex, bestCount[0][0], bestCount[0][1]),best0AttList)

        node.trueChild = createTree(Node(bestAttIndex, 1, possibleIndex, bestCount[1][0], bestCount[1][1]),best1AttList)
        node.negNum = bestCount[0][0] + bestCount[0][1]
        node.posNum = bestCount[1][0] + bestCount[1][1]

    return node


def calculateEntropy(neg, pos):
    if (neg == 0 or pos == 0):
        return 0
    else:
        prob1 = neg / float(neg + pos)
        prob2 = pos / float(neg + pos)
        e1 = math.log2(prob1)
        e2 = math.log2(prob2)

        e = - prob1 * e1 - prob2 * e2
        return e


def checkAccuracy(node, attValues):
    yes = 0
    no = 0
    for x in attValues:
        n = node
        while (n.check == 2):
            i = n.attributeSplitIndex
            if x[0][i] == '0':
                n = n.falseChild
            if x[0][i] == '1':
                n = n.trueChild
        if x[1] == str(n.check):
            # print("yes: ",str(x[0])," class:",str(x[1])," Pred: ",str(n.check))
            yes += 1
        else:
            no += 1
            # print("no: ",str(x[0])," class:",str(x[1])," Pred: ",str(n.check))

    # print(yes)
    # print(no)
    return yes / float(yes + no) * 100


def recurse(node):
    if (node != None):
        return Node(node.attSplitIndex, node.attSplitValue, node.possibleIndex, node.posNum, node.negNum
                    , node.check, node.attributeSplitIndex, recurse(node.falseChild), recurse(node.trueChild))
    else:
        return None


def prune(factor):
    global nodeNum
    global leafNum
    global root
    global bestTree
    global attValues
    global count
    n = int(factor * nodeNum)
    attValuesAccuracy = checkAccuracy(root, attValues)
    currentAccuracy = attValuesAccuracy
    count=0
    while currentAccuracy <= attValuesAccuracy + accThresh or count<15:
        count=count+1
        tree = recurse(root)
        for x in range(n):
            nodeNum = 0
            leafNum = 0
            numCheck(tree)
            k = random.randrange(nodeNum - leafNum)
            count = 0
            pruning(tree, k)
        currentAccuracy = checkAccuracy(tree, attValues)
    bestTree = tree


def pruning(node, k):
    global count
    if (node.check == 2):
        count += 1
        if (count == k and node.attSplitValue != 2):
            # print(node.attributeSplitIndex)
            if (node.negNum > node.posNum):
                node.check = 0
            else:
                if (node.negNum < node.posNum):
                    node.check = 1
                else:
                    node.check = random.randrange(1)
        pruning(node.falseChild, k)
        pruning(node.trueChild, k)


def numCheck(node):
    global nodeNum
    global leafNum
    nodeNum += 1
    if (node.check != 2):
        leafNum += 1
    else:
        numCheck(node.falseChild)
        numCheck(node.trueChild)


def displayTree(node, k):
    if (node.check != 2):
        print(str(comp[node.attSplitIndex]) + " = " + str(node.attSplitValue), end="")
        print(": %d" % node.check)
    else:
        if (node.attSplitValue != 2):
            print(str(comp[node.attSplitIndex]) + " = " + str(node.attSplitValue))
            k += 1
            for i in range(k):
                print("|", end='')
                print(" ", end="")

        displayTree(node.falseChild, k)

        if (node.attSplitValue != 2):
            for i in range(k):
                print("|", end='')
                print(" ", end="")

        displayTree(node.trueChild, k)


def displayAcc(node, str):
    print("\n")
    print(str, "instances : %d" % rowNum)
    print(str, "attributes : %d" % attNum)
    print("Accuracy on the", str, "dataset : %f" % checkAccuracy(node, attValues))


def main(args):
    global root, bestTree, nodeNum, leafNum
    training = args[0]
    validation = args[1]
    test = args[2]
    factor = float(args[3])
    print('\nDecision Tree before Pruning')
    print('------------------------------')
    readData(training)
    tree()
    displayTree(root, 0)

    nodeNum = 0
    leafNum = 0
    numCheck(root)
    print('\nPre-Pruned Accuracy')
    print('--------------------------------')
    print('Total number of nodes : %d' % nodeNum)
    print('Number of leaf nodes : %d' % leafNum)
    displayAcc(root, "Training")
    readData(validation)
    displayAcc(root, "Validation")
    readData(test)
    displayAcc(root, "Testing")

    readData(validation)
    prune(factor)
    print('\n\n\n----------------------------------------------------')
    print('Decision Tree after Pruning with a factor of %0.02f' % factor)
    print('----------------------------------------------------')

    displayTree(bestTree, 0)

    nodeNum = 0
    leafNum = 0
    # displayTree(bestTree,0)
    numCheck(bestTree)
    print('\nPost-Pruned Accuracy')
    print('---------------------------------')
    print('Total number of nodes : %d' % nodeNum)
    print('Number of leaf nodes : %d' % leafNum)
    readData(training)
    displayAcc(bestTree, "Training")
    readData(validation)
    displayAcc(bestTree, "Validation")
    readData(test)
    displayAcc(bestTree, "Testing")

main(sys.argv)