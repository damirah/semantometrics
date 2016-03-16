
import os
import csv
import json
import logging
from logging.config import dictConfig

from contribution.contrib_calculator import ContribCalculator


__author__ = 'Drahomira Herrmannova'
__email__ = 'd.herrmannova@gmail.com'


def setup_logging(default_path='logging.json', default_level=logging.DEBUG):
    """
    Setup logging configuration
    :param default_path:
    :param default_level:
    :return: None
    """
    path = default_path
    log_dir = './log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        config['handlers']['file']['filename'] = \
            os.path.join(log_dir, 'debug.log')
        dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return


if __name__ == '__main__':
    setup_logging()
    main_logger = logging.getLogger(__name__)

    documents = {}
    indices_cited = []
    indices_citing = []

    doc_id = 27 # which document are we calculating contribution for
    data_dir = 'test_data'
    citnet_file = 'citations.tsv'

    main_logger.info('Loading documents')
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            index = os.path.basename(file).replace('.txt', '')
            with open(os.path.join(data_dir, file), 'r') as f:
                documents[index] = f.read()

    main_logger.info('Loading citation network')
    with open(os.path.join(data_dir, citnet_file),'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if int(row[0]) == doc_id:
                indices_cited.append(row[1])
            elif int(row[1]) == doc_id:
                indices_citing.append(row[0])

    contrib_calculator = ContribCalculator()
    contribution = contrib_calculator.contribution(indices_a=indices_cited,
                                    indices_b=indices_citing,
                                    docs=documents)

    main_logger.info('Contribution is {}'.format(contribution))
