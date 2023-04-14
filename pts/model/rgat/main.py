import torch
import torch.nn as nn
import numpy as np
import time

from gluonts.evaluation import make_evaluation_predictions, MultivariateEvaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from pts.model.rgat import RGATEstimator
from pts.modules import StudentTOutput

# from utils.data_utils import ts_gen
import wandb
wandb.init(project="dnri-mod")

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project="dnri-mod")

from parse import parser


def main(args):

    torch.manual_seed(args.seed_torch)

    t0 = time.time()

    dataset = get_dataset(args.dataset, regenerate=False)

    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    if args.target_dim > 0:
        target_dim = args.target_dim
    
    print("Number of nodes: "+str(target_dim))

    train_grouper = MultivariateGrouper(max_target_dim=target_dim)

    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                    max_target_dim=target_dim)

    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)

    estimator = RGATEstimator(
        freq=dataset.metadata.freq,
        target_dim=target_dim,
        prediction_length=dataset.metadata.prediction_length,
        context_length=2*dataset.metadata.prediction_length,
        embedding_dim=args.embedding_dim,
        distr_output=StudentTOutput(target_dim),
        lin_hidden = args.lin_hidden,
        att_hidden = args.att_hidden,
        num_att_heads = args.num_att_heads,
        activation = eval(args.activation),
        activation_att = eval(args.activation_att),
        dropout = args.dropout,
        dropout_att = args.dropout_att,
        lr = args.lr,
        weight_decay = args.weight_decay,
        scaling = eval(args.scaling),
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches_per_epoch,
        trainer_kwargs=dict(
            max_epochs=args.num_epochs,
            accelerator=args.accelerator,
            gpus=1,
            log_every_n_steps=1,
            logger=wandb_logger
        )
    )

    # training
    predictor = estimator.train(
        training_data=dataset_train,
        num_workers=1,
        shuffle_buffer_length=1024,
        cashe_data=True,
    )

    # testing
    forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                                    predictor=predictor,
                                                    num_samples=100)

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                    target_agg_funcs={'sum': np.sum})

    agg_metric, _ = evaluator(tss, forecasts, num_series=len(dataset_test))

    print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
    print("ND: {}".format(agg_metric['ND']))
    print("NRMSE: {}".format(agg_metric['NRMSE']))
    print("MSE: {}".format(agg_metric['MSE']))

    print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
    print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
    print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
    print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))

    wandb.log({"CRPS-Sum":agg_metric['m_sum_mean_wQuantileLoss']})
    wandb.log({"CRPS":agg_metric['mean_wQuantileLoss']})
    wandb.log({"MSE-Sum":agg_metric['m_sum_MSE']})
    wandb.log({"MSE":agg_metric['MSE']})

    print(time.time()-t0)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)