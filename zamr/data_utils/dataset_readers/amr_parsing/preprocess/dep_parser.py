# -*- coding: utf-8 -*-

from pycorenlp import StanfordCoreNLP


class DepParser:
    def __init__(self, url):
        self.nlp = StanfordCoreNLP(url)
        # Do not split the hyphenation
        self.nlp_properties = {
            'annotators': "depparse",
            "tokenize.options": "splitHyphenated=false,normalizeParentheses=false",
            "tokenize.whitespace": True,  # all tokens have been tokenized before
            'ssplit.isOneSentence': True,
            'outputFormat': 'json'
        }

    def get_head(self, text):
        raw_text = " ".join(text)
        parsed = self.nlp.annotate(raw_text.strip(), self.nlp_properties)['sentences'][0]['basicDependencies']
        assert len(text) == len(parsed), "Lengths are note equal."
        dep_head = len(parsed) * [0]
        # Get the head node of each token
        for p in parsed:
            index = p['dependent'] - 1  # avoid ROOT
            head = p['governor'] - 1  # start from 0
            dep_head[index] = head
        return dep_head

    def __call__(self, text):
        return self.get_head(text)
