import argparse
import Index

from sklearn.datasets import load_svmlight_file
import collections
from Index import Index

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import math
import os
import re

import nltk
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English

def createFeatureDefinitionFile(fileName, index):
    f = open(fileName, "w")

    # Write out the feature definition file. Make sure we cast the key to a string so it can be appended.
    for key, value in index.featureIDKeyDictionary.items():
        f.write("(" + str(key) + ", " + value + ")\n")

def parseVocabFile(fileName, index):
    f = open(fileName, "r")

    print("Parsing vocab file...")
    lines = f.readlines()
    for line in lines:
        for word in nltk.word_tokenize(line):
            index.add(word)

    print("Finished parsing vocab file.")
    # print(len(index.termKeyDictionary))

def createTrainingDataFile(fileName, newsDirectory, index, classDictionary, termWeightVal):

    currentDir = os.path.dirname(os.path.realpath(__file__))

    if os.path.isabs(newsDirectory):
        currentDir = newsDirectory
    else:
        # Get correct directory first.
        currentDir += "\\" + newsDirectory

    print(currentDir)

    IDFDictionary = {}

    # This list will store all of dictionaries for each documents terms and their associated TF values.
    listOfDocDictionaries = []
    listOfDocClasses = []

    totalFiles = 0
    for subdir, dirs, files in os.walk(currentDir):
        totalFiles += len(files)

    # Iterate through the news group directory and parse files.
    for subdir, dirs, files in os.walk(currentDir):
        for file in files:
            fullPath = os.path.join(subdir, file)
            currentLineNum = 0
            # Output the current file we are processing
            print(fullPath)
            f = open(fullPath, "r")

            docWordCount = 0
            docDictionary = {}

            # Parse all the lines of each file and check for the key strings
            lines = f.readlines()
            for line in lines:
                if line.startswith("Subject:"):
                    # process this line
                    for word in nltk.word_tokenize(line):
                        featureID = index.lookup(word)
                        # We don't want to add the word if its not in our index. This ignores characters treated as
                        # tokens such as : . , [ etc.
                        if featureID != None:
                            if featureID not in docDictionary:
                                docDictionary[featureID] = 1
                            else:
                                docDictionary[featureID] += 1

                            # Add the docID to IDFDictionary for the current term if it's not in there.
                            currentFileName = file
                            if featureID not in IDFDictionary:
                                IDFDictionary[featureID] = [currentFileName]
                            else:
                                docList = IDFDictionary[featureID]
                                if currentFileName not in docList:
                                    docList.append(currentFileName)
                            docWordCount += 1

                        # Stop words should still be taken into account when counting the total words in a document
                        else:
                            if index.isStopWord(word):
                                docWordCount += 1
                if line.startswith("Lines:"):
                    currentLineNum = re.findall('\d+', line)

            # Process lines starting from bottom of file up based on lines input.
            linesProcessed = 0
            if not currentLineNum:
                # Only one file had a bad input, so here is its line count
                if currentFileName == '39668':
                    currentLineNum = [13]
                elif currentFileName == '104595':
                    currentLineNum = [7]
                elif currentFileName == '15387':
                    currentLineNum = [13]
                elif currentFileName == '59559':
                    currentLineNum = [30]
                elif currentFileName == '60237':
                    currentLineNum = [11]
                elif currentFileName == '75916':
                    currentLineNum = [11]
                elif currentFileName == '75918':
                    currentLineNum = [57]
                elif currentFileName == '76277':
                    currentLineNum = [27]

            # Loop the file lines starting at the bottom, break if we hit our line limit.
            for line in reversed(list(open(fullPath))):
                # if linesProcessed == int(currentLineNum[0]):
                if linesProcessed == int(currentLineNum[0]) - 1:
                    break
                else:
                    # Process this lines terms
                    for word in nltk.word_tokenize(line):
                        featureID = index.lookup(word)
                        if featureID != None:
                            if featureID not in docDictionary:
                                docDictionary[featureID] = 1
                            else:
                                docDictionary[featureID] += 1
                            docWordCount += 1

                            # Add the docID to IDFDictionary for the current term if it's not in there.
                            currentFileName = file
                            if featureID not in IDFDictionary:
                                IDFDictionary[featureID] = [currentFileName]
                            else:
                                docList = IDFDictionary[featureID]
                                if currentFileName not in docList:
                                    docList.append(currentFileName)
                        # Stop words should still be taken into account when counting the total words in a document
                        else:
                            if index.isStopWord(word):
                                docWordCount += 1
                linesProcessed += 1

            # Calculate TF. Here key is the current term in the document and val is the frequency it shows
            # up in the document.
            for key, val in docDictionary.items():
                # docDictionary[key] = val / docWordCount
                docDictionary[key] = math.log2(1 + val)

            od = collections.OrderedDict(sorted(docDictionary.items()))
            # print(od)
            # We are done processing this document's terms, add it to the list and move on to the next file.
            listOfDocDictionaries.append(od)


            className = os.path.basename(subdir)
            if className in classDictionary:
                listOfDocClasses.append(classDictionary[className])
                # writeFile.write(str(classDictionary[className]) + " ")



    # Modify the filename so we know which file is which.
    if termWeightVal == 1:
        fileName += "TFIDF.txt"
    elif termWeightVal == 2:
        fileName += "IDF.txt"
    elif termWeightVal == 3:
        fileName += "TF.txt"

    writeFile = open(fileName, "w")

    # Loops through the list of each document's term dictionary and writes out its class and features.
    for index, dictionary in enumerate(listOfDocDictionaries):
        writeFile.write(str(listOfDocClasses[index]) + " ")

        # Calculate final TF IDF scores for each feature and then output it to the file based on which we care about.
        for featureID in dictionary:
            TF = dictionary[featureID]
            docCountOfCurrentTerm = len(IDFDictionary[featureID])
            IDF = math.log10(totalFiles / docCountOfCurrentTerm)

            TFIDF = TF * IDF
            if termWeightVal == 1:
                writeFile.write(str(featureID) + ":" + str(TFIDF) + " ")
            elif termWeightVal == 2:
                writeFile.write(str(featureID) + ":" + str(IDF) + " ")
            elif termWeightVal == 3:
                writeFile.write(str(featureID) + ":" + str(TF) + " ")
            # writeFile.write(str(featureID) + ":" + str(TFIDF) + " ")

        writeFile.write("\n")


def createClassFile(fileName):
    f = open(fileName, "w")

    # We can hard code these class values and write them out to the file.
    classDictionary = {}
    classDictionary["comp.graphics"] = 1
    classDictionary["comp.os.ms-windows.misc"] = 1
    classDictionary["comp.sys.ibm.pc.hardware"] = 1
    classDictionary["comp.sys.mac.hardware"] = 1
    classDictionary["comp.windows.x"] = 1

    classDictionary["rec.autos"] = 2
    classDictionary["rec.motorcycles"] = 2
    classDictionary["rec.sport.baseball"] = 2
    classDictionary["rec.sport.hockey"] = 2

    classDictionary["sci.crypt"] = 3
    classDictionary["sci.electronics"] = 3
    classDictionary["sci.med"] = 3
    classDictionary["sci.space"] = 3

    classDictionary["misc.forsale"] = 4

    classDictionary["talk.politics.misc"] = 5
    classDictionary["talk.politics.guns"] = 5
    classDictionary["talk.politics.mideast"] = 5

    classDictionary["talk.religion.misc"] = 6
    classDictionary["alt.atheism"] = 6
    classDictionary["soc.religion.christian"] = 6

    for key, val in classDictionary.items():
        f.write("(" + key + ", " + str(val) + ")\n")

    return classDictionary

def tests():
    print("Running tests for feature extraction.")
    # Create index
    index = Index(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))

    # Verify that the Vocab file exists.
    exists = os.path.isfile("VocabList.txt")
    if exists == False:
        print('Cannot find vocab file!!')
        return 0;
    else:
        print("Found Vocab file.")

    parseVocabFile("VocabList.txt", index)

    termCount = len(index.termKeyDictionary)
    print("Term count is: " + str(termCount))
    if termCount == 44985:
        print("Correct number of terms found!!")
    else:
        print("Incorrect nubmer of terms found.")


    print("\n")
    print("Index and TFIDF tests:")
    testIndex = Index(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))

    # Add 2 "Documents"
    terms = nltk.word_tokenize("Jack and Jill went up the hill cause Jack.")
    for term in terms:
        testIndex.add(term)

    terms = nltk.word_tokenize("Jill is very hungry.")
    for term in terms:
        testIndex.add(term)

    print("Jack featureID: " + str(testIndex.lookup("Jack")))
    # TF is log2 of 1 plus the term count.
    TF = math.log2(1 + 1)
    print("Jack TF: " + str(TF))

    # IDF is log10 of total Document count over number of docs the term appears in.
    IDF = math.log10(2 / 1)
    print("Jack IDF: " + str(IDF))

    TFIDF = TF * IDF
    print("Jack TFIDF: " + str(TFIDF))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("NewsGroupDir", help="The input file name containing the documents to parse.")
    parser.add_argument("featureDefFile", help="The name of the file to output the results to.")
    parser.add_argument("classDefFile", help="The name of the file to output the results to.")
    parser.add_argument("trainingDataFile", help="The name of the file to output the results to.")
    args = parser.parse_args()

    # First we parse the list of words in the dataset.
    index = Index(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))
    parseVocabFile("VocabList.txt", index)

    # Create class definition file.
    print("Generating Class file...")
    classDictionary = createClassFile(args.classDefFile)
    print("Finished generating class file.")

    # Create feature definition file
    print("Creating feature defintion file...")
    createFeatureDefinitionFile(args.featureDefFile, index)
    print("Finished writing feature defintion file")


    # Create the training data file
    # 1 = TFIDF, 2 = IDF, 3 = TF
    termWeightVal = 1
    print("Generating training data file...")
    createTrainingDataFile(args.trainingDataFile, args.NewsGroupDir, index, classDictionary, termWeightVal)
    print("Finished generating training data.")

    # Uncomment to run tests
    # tests()
