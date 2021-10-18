# -*- coding: utf-8 -*-

import os

from .annotator import Annotator
from .cleaner import Cleaner
from .recategorizer import Recategorizer
from .anonymizor import Anonymizor
from .senseremover import SenseRemover
from .dep_parser import DepParser


def clean_file_name(file_path):
    file_path_proc = file_path + '.features.input_clean.recategorize.nosense.parsed'
    # Rename
    os.rename(file_path_proc, file_path + '.features.preproc')
    # clear up
    os.remove(file_path + '.features.input_clean')
    os.remove(file_path + '.features.input_clean.recategorize')
    os.remove(file_path + '.features.input_clean.recategorize.nosense')


__all__ = ['Annotator', 'Cleaner', 'Recategorizer', 'Anonymizor', 'SenseRemover', 'DepParser', 'clean_file_name']
