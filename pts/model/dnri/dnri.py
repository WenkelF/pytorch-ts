import torch
import numpy as np

from gluonts.evaluation import make_evaluation_predictions, MultivariateEvaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import TrainingHistory

from pts.model.dnri import DNRIEstimator
from pts.modules import StudentTOutput

from utils import construct_full_graph, construct_expander, construct_expander_fast, construct_random_graph, construct_bipartite_graph

# import wandb
# wandb.init(project="dnri-link-prediction")
# wandb.init(project="dnri")

# from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(project="dnri-link-prediction")

from parse import parser
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(args.seed)

dataset = get_dataset(args.dataset, regenerate=False)

train_list = list(dataset.train)

train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
print("Number of nodes :"+str(target_dim))

# fully connected graph O(N^2) space
if args.graph_constr == "full":
    print("Graph construction: Fully connected graph")
    send_edges, recv_edges = construct_full_graph(target_dim)

# expander graph
if args.graph_constr == "expander":
    print("Graph construction: Expander")
    send_edges, recv_edges = construct_expander(target_dim, args.graph_density)

# expander graph (fast)
if args.graph_constr == "expander_fast":
    print("Graph construction: Expander")
    send_edges, recv_edges = construct_expander_fast(target_dim, args.graph_density)
        
# sparse random graph
if args.graph_constr == "random":
    print("Graph construction: Random graph")
    send_edges, recv_edges = construct_random_graph(target_dim, args.graph_density)
   
# sparse bipartite graph
if args.graph_constr == "bipartite":
    print("Graph construction: Bipartite graph")
    send_edges, recv_edges = construct_bipartite_graph(target_dim, args.graph_density)

edges = [send_edges, recv_edges]

torch.save(edges, './checkpoints/edges.pt')


dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)

from custom_callbacks import ReplaceEdgeCallback, EdgeUsageCallback, EpochCallback
from pytorch_lightning.callbacks import ModelCheckpoint

# callback_list = [EpochCallback(), ModelCheckpoint(dirpath='./checkpoints', filename='{epoch}', save_top_k=-1, every_n_epochs=2)]
callback_list = []

if args.link_prediction == 1:
    print("Auxiliary link prediction...")
    callback_list.append(ReplaceEdgeCallback(args.pre_epochs, args.post_epochs, args.mod_freq, args.num_mods))

estimator = DNRIEstimator(
    freq=dataset.metadata.freq,
    context_length=2*dataset.metadata.prediction_length,
    prediction_length=dataset.metadata.prediction_length,
    
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    distr_output=StudentTOutput(int(dataset.metadata.feat_static_cat[0].cardinality)),
    
    # DNRI hyper-params
    mlp_hidden_size=args.hidden_dim_mlp,
    decoder_hidden=args.hidden_dim_dec,
    rnn_hidden_size=args.hidden_dim_rnn,
    edges=edges,
    link_prediction = args.link_prediction,
    num_layers=args.num_layers,
    
    # training hyperparams
    batch_size=args.batch_size,
    num_batches_per_epoch=args.num_batches_per_epoch,
    # trainer_kwargs=dict(max_epochs=args.num_epochs, accelerator='gpu', gpus=1, callbacks=callback_list, logger=wandb_logger),
    trainer_kwargs=dict(max_epochs=args.num_epochs, accelerator='gpu', gpus=1, callbacks=callback_list),
)

# training
predictor = estimator.train(
    training_data=dataset_train,
    num_workers=4,
    shuffle_buffer_length=1024
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

# wandb.log({"CRPS":agg_metric['mean_wQuantileLoss']})
# wandb.log({"ND":agg_metric['ND']})
# wandb.log({"NRMSE":agg_metric['NRMSE']})
# wandb.log({"MSE":agg_metric['MSE']})

# wandb.log({"CRPS-Sum":agg_metric['m_sum_mean_wQuantileLoss']})
# wandb.log({"ND-Sum":agg_metric['m_sum_ND']})
# wandb.log({"NRMSE-Sum":agg_metric['m_sum_NRMSE']})
# wandb.log({"MSE-Sum":agg_metric['m_sum_MSE']})