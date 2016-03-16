
# standard python modules
import logging

# installed modules
import numpy

# custom modules
from contribution.dist_calculator import DistCalculator

__author__ = 'Drahomira Herrmannova'
__email__ = 'd.herrmannova@gmail.com'


class ContribCalculator(object):
    """
    Class for calculating publication contribution according to the equation
    presented in http://www.dlib.org/dlib/november14/knoth/11knoth.html.
    """

    def __init__(self, distances=None):
        """
        Constructor sets up logging and initialises DistCalculator class for
        calculating semantic distance between documents. To speed up
        the calculation the class can be passed a map with some or all
        pre-calculated distances.
        :param distances: map of document distances in the format
                          {index1: {index2: distance}}
        """
        self.logger = logging.getLogger(__name__)
        self.dist_calculator = DistCalculator()
        self.calculated_distances = distances if distances is not None else {}

    def get_calculated_distances(self):
        """
        :return: map in the format {index1: {index2: distance}}
        """
        return self.calculated_distances

    def _get_document_text(self, doc_map, idx):
        """
        Retrieve document text from doc_map, return empty string in case
        idx does not exist in the map or in case the text is empty.
        :param doc_map: map in the format {index: document_text}
        :param idx: index of the document text to be retrieved
        :return: string
        """
        if idx not in doc_map:
            return ''
        doc_text = doc_map[idx]
        return '' if doc_text is numpy.nan or doc_text is None else doc_text

    def _add_distance(self, idx1, idx2, distance):
        """
        Add distance between documents idx1 and idx2 to the map of calculated
        distances.
        :param idx1: ID of document 1
        :param idx2: ID of document 2
        :param distance: distance between the documents
        :return: None
        """
        if idx1 not in self.calculated_distances:
            self.calculated_distances[idx1] = {}
        if idx2 not in self.calculated_distances:
            self.calculated_distances[idx2] = {}
        self.calculated_distances[idx1][idx2] = distance
        self.calculated_distances[idx2][idx1] = distance
        return

    def _pairwise_distances(self, indices_a, indices_b, idx_text_map):
        """
        Takes two sets of indices and a map of the document texts and calculates
        pairwise distances between the two sets. The result is a list with
        the distances.
        :param indices_a: list of document indices
        :param indices_b: list of document indices
        :param idx_text_map: map in the format {index: document_text}
        :return: list of distances between documents in indices_a and indices_b
        """
        self.logger.debug('Calculating distances for indices {0} and {1}'
                          .format(indices_a, indices_b))
        distances = []
        for pair in [[x, y] for x in indices_a for y in indices_b]:
            self.logger.debug('Calculating distance for IDs {0}'.format(pair))
            if pair[0] == pair[1]:
                self.logger.debug('Identical indices, skipping')
                continue
            if pair[0] in self.calculated_distances \
                    and pair[1] in self.calculated_distances[pair[0]]:
                dist = self.calculated_distances[pair[0]][pair[1]]
                self.logger.info('Reusing distance {0}'.format(dist))
                distances.append(dist)
            else:
                self.logger.info('Calculating distance')
                d1 = self._get_document_text(idx_text_map, pair[0])
                d2 = self._get_document_text(idx_text_map, pair[1])
                self.logger.debug('Got two abstracts, calculating distance')
                dist = self.dist_calculator.document_distance(d1, d2)
                if dist is not None:
                    distances.append(dist)
                    self._add_distance(pair[0], pair[1], dist)
        self.logger.debug('Done calculating distances, returning')
        return distances

    def _mean_distance(self, indices_a, indices_b, docs):
        """
        :param indices_a:
        :param indices_b:
        :param docs:
        :return:
        """
        distances = self._pairwise_distances(indices_a, indices_b, docs)
        if len(distances) == 0:
            self.logger.warn('Could not calculate distances')
            return None
        else:
            mean_distance = sum(distances) / len(distances)
            self.logger.info('Mean distance is {0}'.format(mean_distance))
            return mean_distance

    def contribution(self, indices_a, indices_b, docs):
        """
        Calculates contribution of a publication based on the equation presented
        in http://www.dlib.org/dlib/november14/knoth/11knoth.html
        :param indices_a: Indices of documents citing the publication for which
                          contribution should be calculated
        :param indices_b: Indices of documents cited by the publications for
                          which contribution should be calculated
        :param docs: map in the format {index: document_text} with
                     full-texts/abstracts of the citing and cited documents
        :return: contribution value
        """

        if len(indices_a) == 0 or len(indices_b) == 0:
            self.logger.info('No citing or cited docs, returning None')
            return None

        if len(indices_a) == 1 or len(indices_b) == 1:
            self.logger.info('Only one citing or cited paper, setting '
                             'adjustment parameters to 1')
            overline_a = 1
            overline_b = 1
        else:
            self.logger.info('More than one citing and cited paper, '
                             'calculating adjustment parameters')
            overline_a = self._mean_distance(indices_a, indices_a, docs)
            overline_b = self._mean_distance(indices_b, indices_b, docs)

        if (overline_a <= 0 or overline_a > 1 or
                    overline_b <= 0 or overline_b > 1):
            self.logger.warn('One of the adjustment parameters is not '
                             'in the interval (0, 1]')
            return None

        self.logger.info('Calculating inter group distance')
        mean_distance = self._mean_distance(indices_a, indices_b, docs)

        if mean_distance < 0 or mean_distance > 1:
            self.logger.warn('Mean distance is out of the interval [0, 1]')
            return None

        self.logger.debug('Calculating contribution')
        adjust = overline_b / overline_a
        self.logger.info('Adjusting by {0}, mean distance is {1}'
                         .format(adjust, mean_distance))
        contribution = adjust * mean_distance
        self.logger.info('Done, contribution is {0}'.format(contribution))
        return contribution
