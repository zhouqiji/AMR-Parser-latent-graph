# -*- coding: utf-8 -*-

import re
import nltk

from zamr.utils import logging
from zamr.data_utils.dataset_readers.amr_parsing.io import AMRIO

logger = logging.init_logger()

class SenseRemover:
    """
    Remove sense numbers of AMR node instances
    """

    def __init__(self, node_utils):
        self.node_utils = node_utils
        self.stemmer = nltk.stem.SnowballStemmer('english').stem

        self.removed_instance_count = 0
        self.amr_instance_count = 0
        self.restore_count = 0
        self.not_removed_instances = set()

    def remove_file(self, file_path):
        for i, amr in enumerate(AMRIO.read(file_path), 1):
            if i % 1000 == 0:
                logger.info('Processed {} examples.'.format(i))
            self.remove_graph(amr)
            yield amr

    def remove_graph(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            if node.copy_of is not None:
                continue
            instance = node.instance
            lemmas = self.map_instance_to_lemmas(instance)
            lemma = self.find_corresponding_lemma(instance, lemmas, amr)
            if lemma is None:
                lemma = self.remove_sense(instance)
            self.update_graph(graph, node, instance, lemma)

    def map_instance_to_lemmas(self, instance):
        """
        Get the candidate lemmas which can be used to represent the instance.
        """
        # Make sure it's a string and not quoted.
        if not (isinstance(instance, str) and not re.search(r'^".*"$', instance)):
            instance = str(instance)
        if re.search(r'-\d\d$', instance):  # frame
            lemmas = self.node_utils.get_lemmas(instance)
        else:
            lemmas = [instance]
        return lemmas

    def find_corresponding_lemma(self, instance, lemmas, amr):
        self.amr_instance_count += 1
        input_lemma = None
        for lemma in lemmas:
            if lemma in amr.lemmas:
                input_lemma = lemma
                break

        # Make sure it can be correctly restored.
        if input_lemma is not None:
            restored_frame = self.node_utils.get_frames(input_lemma)[0]
            if restored_frame != instance:
                input_lemma = None

        if input_lemma is None:
            self.not_removed_instances.add(instance)
        else:
            self.removed_instance_count += 1

        return input_lemma

    def remove_sense(self, instance):
        instance_lemma = re.sub(r'-\d\d$', '', str(instance))
        restored = self.node_utils.get_frames(instance_lemma)[0]
        if restored == instance:
            return instance_lemma
        return instance

    def update_graph(self, graph, node, old, new):
        if new is not None:
            graph.replace_node_attribute(node, 'instance', old, new)
            self.try_restore(str(old), new)
        else:
            self.try_restore(old, old)

    def try_restore(self, old, new):
        _old = self.node_utils.get_frames(new)[0]
        self.restore_count += int(old == _old)

    def reset_statistics(self):
        self.removed_instance_count = 0
        self.amr_instance_count = 0
        self.restore_count = 0
        self.no_removed_instances = set()

    def print_statistics(self):
        print('sense remove rate: {}% ({}/{})'.format(
            self.removed_instance_count / self.amr_instance_count,
            self.removed_instance_count, self.amr_instance_count))
        print('restore rate: {}% ({}/{})'.format(
            self.restore_count / self.amr_instance_count,
            self.restore_count, self.amr_instance_count))
        print('size of not removed lemma set: {}'.format(len(self.not_removed_instances)))
