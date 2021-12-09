# Copyright (c) 2020, Ioana Bica

import os
import argparse
import logging

from CRN_encoder_evaluate import test_CRN_encoder
from CRN_decoder_evaluate import test_CRN_decoder
# from utils.cancer_simulation import get_cancer_sim_data
from Transform import get_data


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=2, type=int)
    parser.add_argument("--radio_coeff", default=2, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--model_name", default="crn_test_2")
    parser.add_argument("--b_encoder_hyperparm_tuning", default=False)
    parser.add_argument("--b_decoder_hyperparm_tuning", default=False)
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    pickle_map = get_data(num_month=16, id_entry=8, model_root=args.results_dir)

    encoder_model_name = 'encoder_' + args.model_name
    encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)

    models_dir = '{}/crn_models'.format(args.results_dir)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    rmse_encoder = test_CRN_encoder(pickle_map=pickle_map, models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning)

    decoder_model_name = 'decoder_' + args.model_name
    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

    """
    The counterfactual test data for a sequence of treatments in the future was simulated for a 
    projection horizon of 5 timesteps. 

    """

    max_projection_horizon = 5
    projection_horizon = 5

    rmse_decoder = test_CRN_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
                                    projection_horizon=projection_horizon,
                                    models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    decoder_model_name=decoder_model_name,
                                    decoder_hyperparams_file=decoder_hyperparams_file,
                                    b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning)

    logging.info("Chemo coeff {} | Radio coeff {}".format(args.chemo_coeff, args.radio_coeff))
    print("RMSE for one-step-ahead prediction.")
    print(rmse_encoder)

    print("Results for 5-step-ahead prediction.")
    print(rmse_decoder)