# -*- coding: utf-8 -*-

import json
import yaml
import copy

from zamr.utils import logging
from .algorithms import dict_merge

logger = logging.init_logger()


class Config:
    """
    Loading configs
    """

    def __init__(self, configs):
        self.configs = configs

    def __eq__(self, other):
        if not isinstance(other, Config):
            logger.info("The configs you compare is not and instance of Config. ({} != {})".format(
                type(self), type(other)
            ))
            return False

        this_flat_params = self.as_flat_dict()
        other_flat_params = other.as_flat_dict()

        if len(this_flat_params) != len(other_flat_params):
            logger.info('The numbers of parameters are different: {} != {}'.format(
                len(this_flat_params),
                len(other_flat_params)
            ))
            return False

        same = True
        for k, v in this_flat_params.items():
            if k == 'environment.recover':
                continue
            if k not in other_flat_params:
                logger.info('The parameter "{}" is not specified.'.format(k))
                same = False
            elif other_flat_params[k] != v:
                logger.info('The values of "{}" not not the same: {} != {}'.format(
                    k, v, other_flat_params[k]
                ))
                same = False
        return same

    def __getitem__(self, item):
        if item in self.configs:
            return self.configs[item]
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.configs[key] = value

    def __delitem__(self, key):
        del self.configs[key]

    def __iter__(self):
        return iter(self.configs)

    def __len__(self):
        return len(self.configs)

    def items(self):
        return self.configs.items()

    def get(self, key, default=None):
        return self.configs.get(key, default)

    def as_flat_dict(self):
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value

        recurse(self.configs, [])
        return flat_params

    def to_file(self, output_json_file):
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(self.configs, f, indent='\t')

    @classmethod
    def from_file(cls, params_file_list):
        params_file_list = params_file_list.split(",")
        params_dict = {}
        for params_file in params_file_list:
            with open(params_file, encoding='utf-8') as f:
                if params_file.endswith('.yaml'):
                    dict_merge.dict_merge(params_dict, yaml.safe_load(f))
                elif params_file.endswith('.json'):
                    params_dict = json.load(f)
                else:
                    raise NotImplementedError
        return cls(params_dict)

    def __repr__(self):
        return json.dumps(self.configs, indent=2)

    def duplicate(self) -> 'Config':
        """
        Uses ``copy.deepcopy()`` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return Config(copy.deepcopy(self.configs))


def remove_pretrained_embedding_params(configs):
    def recurse(parameters, key):
        for k, v in parameters.items():
            if key == k:
                parameters[key] = None
            elif isinstance(v, dict):
                recurse(v, key)

    recurse(configs, 'pretrained_file')
