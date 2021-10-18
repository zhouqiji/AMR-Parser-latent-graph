# -*- coding: utf-8 -*-

import os
import re

import torch

from zamr import models as Models

from zamr.utils import logging
from zamr.utils.configs import Config, remove_pretrained_embedding_params

from zamr.data_utils.dataset_builder import dataset_from_params, iterator_from_params
from zamr.data_utils.vocabulary import Vocabulary
from zamr.training.trainer import Trainer
from zamr.utils import environment
from zamr.utils.checks import ConfigurationError
from zamr.utils.archival import CONFIG_NAME, _DEFAULT_WEIGHTS, archive_model

logger = logging.init_logger()


class Train:
    """A callable class for training the model."""

    @staticmethod
    def add_subparser(name, parser):
        subparser = parser.add_parser(
            name, help='Train the model.'
        )
        subparser.add_argument('config_file', help='Path of config files.',
                               default='./configs/amr_2.0-hybrid.yaml')
        return subparser

    def __call__(self, args):
        params = Config.from_file(args.config_file)
        logger.info(params)
        # training.
        self.train_model(params=params)

    def train_model(self, params: Config):
        """
        Train the the model with specified configs.
        """
        # set up the environment.
        environment_params = params['environment']
        environment.set_seed(environment_params)
        self.create_serialization_dir(params)
        environment.prepare_global_logging(environment_params)
        environment.check_for_gpu(environment_params)
        if environment_params['gpu']:
            device = torch.device('cuda:{}'.format(environment_params['cuda_device']))
            environment.occupy_gpu(device)
        else:
            device = torch.device('cpu')
        params['trainer']['device'] = device

        # load data
        data_params = params['data']
        dataset = dataset_from_params(data_params)
        train_data = dataset.get('train')
        dev_data = dataset.get('dev')

        # create vocabulary and iterator
        vocab_params = params.get('vocab', {})
        vocab = Vocabulary.from_instances(instances=train_data, **vocab_params)
        # save the vocab file
        vocab.save_to_files(os.path.join(environment_params['serialization_dir'], 'vocabulary'))

        train_iterator, dev_iterator, test_iterator = iterator_from_params(vocab, data_params['iterator'])

        # build model
        model_params = params['model']
        model = getattr(Models, model_params['model_type']).from_params(vocab, model_params)
        logger.info(model)

        # train the model

        # stop the gradient of non-trainable parameters
        trainer_params = params['trainer']
        no_grad_regexes = trainer_params['no_grad']
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = environment.get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are frozen (non gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are tunable (gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        trainer = Trainer.from_params(model, train_data, dev_data, train_iterator, dev_iterator, trainer_params,
                                      graph_type=params['model']['graph_encoder']['graph_type'])

        serialization_dir = trainer_params['serialization_dir']

        try:
            _ = trainer.train()
        except KeyboardInterrupt:
            # if an epoch completed, try to create a model archive.
            if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
                logger.info(
                    "Training interrupted by the user. Attempting to create a model"
                    "archive using the current best epoch weights.")
                archive_model(serialization_dir)

            raise

        # Save the results
        archive_model(serialization_dir)

        logger.info("Loading the best weights.")
        best_model_state_path = os.path.join(serialization_dir, 'best.th')
        best_model_state = torch.load(best_model_state_path)
        best_model = model

        if not isinstance(best_model, torch.nn.DataParallel):
            best_model_state = {re.sub(r'^module\.', '', k): v for k, v in best_model_state.items()}
        best_model.load_state_dict(best_model_state)
        return best_model

    @staticmethod
    def create_serialization_dir(params: Config) -> None:
        """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.
    Parameters
    ----------
    """

        serialization_dir = params['environment']['serialization_dir']
        recover = params['environment']['recover']
        if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
            if not recover:
                raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                         f"not empty. Specify --recover to recover training from existing output.")

            logger.info(f"Recovering from prior training at {serialization_dir}.")

            recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
            if not os.path.exists(recovered_config_file):
                raise ConfigurationError("The serialization directory already exists but doesn't "
                                         "contain a config.json. You probably gave the wrong directory.")
            else:
                loaded_params = Config.from_file(recovered_config_file)

                if params != loaded_params:
                    raise ConfigurationError("Training configuration does not match the configuration we're "
                                             "recovering from.")

                # In the recover mode, we don't need to reload the pre-trained embeddings.
                remove_pretrained_embedding_params(params)
        else:
            if recover:
                raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                         "does not exist.  There is nothing to recover from.")
            os.makedirs(serialization_dir, exist_ok=True)
            params.to_file(os.path.join(serialization_dir, CONFIG_NAME))
