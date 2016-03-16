
# standard python modules
import logging
import string

# installed modules
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


__author__ = 'Drahomira Herrmannova'
__email__ = 'd.herrmannova@gmail.com'


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


def normalize(text):
    return stem_tokens(nltk.word_tokenize(
        text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')


class DistCalculator:
    """
    Class for calculating semantic distance of two texts. It uses NLTK library
    for the calculation. The class has only one method, which accepts two texts
    d1 and d2 and returns their semantic distance. The distance is calculated as
    1-sim(d1, s2), the similarity method used is cosine similarity calculated
    on TFIDF document vectors.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Constructor only sets up logging
        :param log_level: default is logging.INFO
        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

    def document_distance(self, d1, d2):
        """
        :param d1: text (full-text/abstract) of document 1
        :param d2: text (full-text/abstract) of document 2
        :return: distance of the passed documents (value between 0 and 1) or
                 None in case the distance couldn't be calculated (e.g. one of
                 the documents was empty)
        """
        self._logger.debug('Calculating distance of documents'
                          '\n\nD1: {0}\n\nD2: {1}'.format(d1, d2))
        if not d1 or not d2:
            self._logger.warn('One of the texts was empty')
            return None
        documents = [d1, d2]
        # the method vectorizer.fit_transform will:
        # 1. remove punctuation
        # 2. tokenize the texts
        # 3. stem the tokens
        # 4. remove stop words
        # 5. convert texts to vectors
        try:
            tfidf = vectorizer.fit_transform(documents)
        except ValueError:
            self._logger.warn('Empty vocabulary')
            return None
        # no need to normalise separately, since Vectorizer returns
        # normalised tf-idf
        pairwise_similarity = tfidf * tfidf.T
        distance = 1 - pairwise_similarity.A[0][1]
        self._logger.debug('Distance is {0}'.format(distance))
        if distance < 0 or distance > 1:
            self._logger.warn('Incorrect distance: %s', distance)
            return None
        else:
            return distance
