# -*- coding: utf-8 -*-

import os

from zamr.utils import logging
from zamr.data_utils.dataset_readers.amr_parsing.preprocess import Annotator, Cleaner, Recategorizer, Anonymizor, \
    SenseRemover, DepParser, clean_file_name
from zamr.data_utils.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

logger = logging.init_logger()


class Preprocess:
    """A callable class for pre-processing."""

    @staticmethod
    def add_subparser(name, parser):
        subparser = parser.add_parser(
            name, help='Pre-processing data_utils.'
        )

        # For annotating
        subparser.add_argument('files', nargs='+', help='AMR files to annotate.')
        subparser.add_argument('--compound_file', help='Compound file.',
                               default='data/AMR/amr_2.0_utils/joints.txt')

        # For re-categorization
        subparser.add_argument('--util_dir', help='AMR utils files (2.0 or 1.0)',
                               default='data/AMR/amr_2.0_utils/')
        subparser.add_argument('--amr_train_file', help='Amr train files (.features.input_clean)')
        subparser.add_argument('--build_utils', action='store_true')

        return subparser

    def __call__(self, args):

        # Annotating tokens
        print("Annotating...")
        from zamr.data_utils.dataset_readers.amr_parsing.io import AMRIO
        annotator = Annotator('http://localhost:9000', args.compound_file)
        for file_path in args.files:
            logger.info("Annotating {}".format(file_path))
            with open(file_path + '.features', 'w', encoding='utf-8') as f:
                for i, amr in enumerate(AMRIO.read(file_path), 1):
                    if i % 1000 == 0:
                        logger.info('{} processed.'.format(i))
                    annotation = annotator(amr.sentence)
                    amr.tokens = annotation['tokens']
                    amr.lemmas = annotation['lemmas']
                    amr.pos_tags = annotation['pos_tags']
                    amr.ner_tags = annotation['ner_tags']
                    AMRIO.dump([amr], f)

        # Clean inputs
        # Normalizing input data such as date, joint name etc. and correct wrong amr tokens
        print("Cleaning inputs...")
        cleaner = Cleaner()
        for file_path in args.files:
            file_path = file_path + '.features'
            logger.info("Cleaning {}".format(file_path))
            with open(file_path + '.input_clean', 'w', encoding='utf-8') as f:
                for amr in AMRIO.read(file_path):
                    cleaner(amr)
                    f.write(str(amr) + '\n\n')

        # Re-categorizing sub-graph
        # Re-categorization of sub-graphs which merge verbose nodes in AMR. (Date, Polarity, wiki...)
        print("Re-categorizing train and dev data...")
        recategorizer = Recategorizer(
            train_data=args.amr_train_file,
            build_utils=args.build_utils,
            util_dir=args.util_dir
        )
        for file_path in args.files:
            # Only categorize train and dev data
            if 'test' in file_path:
                continue
            file_path = file_path + '.features.input_clean'
            logger.info("Re-categorizing  {}".format(file_path))
            with open(file_path + '.recategorize', 'w', encoding='utf-8') as f:
                for amr in recategorizer.recategorize_file(file_path):
                    f.write(str(amr) + '\n\n')

        # Anonymize test data
        print("Anonymizing test data...")
        anonymizor = Anonymizor.from_json(
            os.path.join(args.util_dir, 'text_anonymization_rules.json')
        )
        for file_path in args.files:
            # Only Anonymize test data
            if 'test' in file_path:
                file_path = file_path + '.features.input_clean'
                logger.info("Re-categorizing  {}".format(file_path))
                with open(file_path + '.recategorize', 'w', encoding='utf-8') as f:
                    for amr in AMRIO.read(file_path):
                        amr.abstract_map = anonymizor(amr)
                        f.write(str(amr) + '\n\n')
            else:
                continue

        # Removing senses
        print("Removing senses...")
        node_utils = NU.from_json(args.util_dir, 0)
        # remove sense under some conditions
        remover = SenseRemover(node_utils)
        for file_path in args.files:
            file_path = file_path + '.features.input_clean.recategorize'
            logger.info("Removing Sense in file:  {}".format(file_path))
            with open(file_path + '.nosense', 'w', encoding='utf-8') as f:
                for amr in remover.remove_file(file_path):
                    f.write(str(amr) + '\n\n')
            remover.reset_statistics()

        # ============================================================
        print("Get head(governor) of tokens...")
        dep_parser = DepParser('http://localhost:9000')
        for file_path in args.files:
            file_path = file_path + '.features.input_clean.recategorize.nosense'
            logger.info("DepParsing {}".format(file_path))
            with open(file_path + '.parsed', 'w', encoding='utf-8') as f:
                for i, amr in enumerate(AMRIO.read(file_path), 1):
                    if i % 1000 == 0:
                        logger.info('{} processed.'.format(i))
                    dep_heads = dep_parser(amr.tokens)
                    amr.dep_heads = dep_heads
                    AMRIO.dump([amr], f)

                #  Clearing up files

        print("Clear up preprocessed files...")
        for file_path in args.files:
            clean_file_name(file_path)

    logger.info('Finishing Pre-processing!')
