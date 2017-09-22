import os
import sys
import re
import math

stopWords = ["", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
          "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
          "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
          "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
          "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
          "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
          "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
          "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
          "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
          "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
          "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
          "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
          "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
          "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

accuracyList = []

def trainData(filePath):
    dictionary = dict()
    temp = 0
    for name in os.scandir(filePath):
        if name.is_file():
            fileName = filePath+"\\"+name.name
            file = open(fileName)
            index = 0
            noOfLines = 0
            for row in file:
                if row.__contains__("Lines: "):
                    noOfLines = int(row.split("Lines:")[1])
                    continue
                if (noOfLines != 0 and index <= noOfLines):
                    index += 1
                    if row != "\n":
                        temp += calculate(row, dictionary)
                elif (noOfLines != 0):
                    break
    return dictionary, temp


def calculate(row, dictionery):
    data = []
    global stopWords
    rowData = re.sub(r"[^a-zA-Z0-9]+", " ", row)
    rowData = rowData.split(" ")
    for each in rowData:
        if (each.lower() not in stopWords):
            data.append(each.lower())

    for word in data:
        if word not in dictionery:
            dictionery[word] = 1
        else:
            dictionery[word] = dictionery[word] + 1

    return len(data)


def testData(data, struct, filePath, fileCount, fileName):
    print("File %d : %s" % (fileCount+1, fileName))
    right = 0
    wrong = 0
    global accuracyList
    for name in os.scandir(filePath):
        if name.is_file():
            fileName = filePath+"\\"+name.name
            file = open(fileName)
            index = 0
            noOfLines = 0
            words = []
            for row in file:
                if row.__contains__("Lines: "):
                    noOfLines = int(row.split("Lines:")[1])
                    continue
                if (noOfLines != 0 and index <= noOfLines):
                    index += 1
                    if row != "\n":
                        words = filter(row, words)
                elif (noOfLines != 0):
                    break
            right, wrong = classify(words, data, struct, fileCount, right, wrong)
    acc = right / (right + wrong) * 100
    accuracyList.append(acc)
    print("Accuracy: %0.2f%%" % acc)
    print("------------------")


def filter(rowData, words):
    global stopWords
    rowData = re.sub(r"[^a-zA-Z0-9]+", " ", rowData)
    rowData = rowData.split(" ")
    for each in rowData:
        if (each.lower() not in stopWords):
            words.append(each.lower())
    return words



def classify(words, data, struct, fileCount, right, wrong):
    probabilities = []
    prior = []
    temp = 0
    for i in range(len(struct)):
        temp += struct[i]

    for i in range(len(data)):
        prior.append(struct[i] / temp)

    i = 0
    for o in data:
        likelihood = 0.0
        for word in words:
            likelihood += math.log((o.get(word, 0.01)) / (struct[i]))

        likelihood += math.log(prior[i])
        probabilities.append(likelihood)
        i += 1
    if probabilities.index(max(probabilities)) == fileCount:
        right += 1
    else:
        wrong += 1
    return right, wrong


def main(args):
    trainPath=args[1]
    testPath=args[2]
    # trainPath = "20news-bydate-train"
    # testPath = "20news-bydate-test"
    train = ["alt.atheism", "comp.graphics", "misc.forsale", "rec.autos", "sci.crypt", "soc.religion.christian", "talk.politics.guns"]
    data = []
    struct = []
    sum = 0
    print("-----------------------")
    print("Training in Progress...")
    print("-----------------------")
    for i in range(len(train)):
        print(i+1,train[i])
        dictionery, temp = trainData(trainPath + "\\" + train[i])
        data.append(dictionery)
        struct.append(temp)
        print("%d / %d Completed" % (i+1,len(train)))
    print("-----------------------")
    print("Testing in Progress...")
    print("-----------------------")
    test = ["alt.atheism", "comp.graphics", "misc.forsale", "rec.autos", "sci.crypt", "soc.religion.christian", "talk.politics.guns"]
    for i in range(len(test)):
        testData(data, struct, testPath + "\\" + test[i], i, test[i])

    for u in range(len(accuracyList)):
        sum+=accuracyList.__getitem__(u)

    overallAccuracy = sum / len(accuracyList)
    print("--------------------------------------")
    # print(sum,len(accuracyList))
    print("Overall Accuracy of Testing Data: %0.2f%%" % overallAccuracy)
    print("--------------------------------------")

# main()
main(sys.argv)
