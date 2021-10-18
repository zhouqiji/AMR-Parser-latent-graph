# -*- coding: utf-8 -*-


class Evaluate:
    """A callable class for evaluating the model."""

    @staticmethod
    def add_subparser(name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the model.'
        )
        return subparser

    def __call__(self, *args, **kwargs):
        print("Evaluating....")
