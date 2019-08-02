import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import numpy


class Index:
    """ Inverted index datastructure """

    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer

        # Dictionary with terms as keys and their associated feature ID's
        self.termKeyDictionary = {}

        # Dictionary with featureID's as keys and their associated term
        self.featureIDKeyDictionary = {}

        #self.vector = numpy.zeros(len(self.items))
        self.documents = {}
        self.uniqueFeatureID = 1
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    # Returns feature ID of the word in the dictionary, or None if it does not exist.
    def lookup(self, word):
        """
        Lookup a word in the index
        """
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)

        # Return the id of the word in dictionary.
        if word in self.termKeyDictionary:
            return self.termKeyDictionary[word]

    def add(self, word):
        # Skip stop words
        word = word.lower()
        if word not in self.stopwords:
            stemWord = self.stemmer.stem(word)
            if stemWord in self.termKeyDictionary:
                existingFeatureID = self.termKeyDictionary[stemWord]
                # increment count
            else:
                self.featureIDKeyDictionary[self.uniqueFeatureID] = stemWord
                self.termKeyDictionary[stemWord] = self.uniqueFeatureID
                self.uniqueFeatureID += 1

    def isStopWord(self, word):
        word = word.lower()
        if word in self.stopwords:
            return True
        else:
            return False


def tests():
    # Tests to verify that the index is working correctly.
    index = Index(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))

    index.add('Industrial Disease')
    index.add('With')
    index.add('BAlls')
    index.add('Ball')
    index.add('Private Investigations')
    index.add('So Far Away')
    index.add('Twisting by the Pool')
    index.add('Skateaway')
    index.add('Walk of Life')
    index.add('Romeo and Juliet')
    index.add('Tunnel of Love')
    index.add('Money for Nothing')
    index.add('Sultans of Swing')

    # Index lookup tests.
    print("Index Lookup tests:")
    print(index.lookup('Industrial Disease'))
    print(index.lookup('Balling'))

    # Expect None here since this term does not exist.
    print(index.lookup('Missing'))

    print("\n")
    # Test for that stop words work.
    print("Stop Word Tests:")
    print(index.isStopWord("And") == True)
    print(index.isStopWord("or") == True)
    print(index.isStopWord("Facts") == False)

if __name__ == '__main__':
    tests()
