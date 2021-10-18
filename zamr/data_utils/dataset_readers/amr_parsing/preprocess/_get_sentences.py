# -*- coding: utf-8 -*-

import os
import argparse

parser = argparse.ArgumentParser(description="Extract AMR sentence.")

parser.add_argument('files', nargs='+', help='AMR files')
parser.add_argument('--output_dir', default="./")

args = parser.parse_args()

if __name__ == '__main__':
    from zamr.data_utils.dataset_readers.amr_parsing.io import AMRIO

    for file_path in args.files:
        print(file_path)
        output_file = file_path.split('/')[-1]
        with open(args.output_dir + output_file + '.sents', 'w', encoding='utf-8') as f:
            for i, amr in enumerate(AMRIO.read(file_path), 1):
                f.write(str(i) + " : " + str(amr.sentence) + '\n\n')
