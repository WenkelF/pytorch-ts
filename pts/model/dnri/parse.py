import argparse

parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default="solar_nips", help='Dataset.')

# model
parser.add_argument('--seed', type=int, default=0, help='numpy.random seed.')
parser.add_argument('--seed_torch', type=int, default=0, help='torch seed.')
parser.add_argument('--graph_density', type=float, default=0.05, help='Density of graph.')
parser.add_argument('--graph_constr', type=str, default="expander", help='Graph construction method.')
parser.add_argument('--num_layers', type=int, default=1, help='Number message passing layers.')

# training

parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--num_batches_per_epoch', type=int, default=100, help='Number of batches per epochs.')
parser.add_argument('--hidden_dim_mlp', type=int, default=16, help='Number of neurons in MLPs.')
parser.add_argument('--hidden_dim_dec', type=int, default=16, help='Number of neurons in decoder.')
parser.add_argument('--hidden_dim_rnn', type=int, default=16, help='Number of neurons in RNNs.')
parser.add_argument('--link_prediction', type=int, default=0, help='Link prediction.')
parser.add_argument('--pre_epochs', type=int, default=5, help='Number of epochs before starting link prediction.')
parser.add_argument('--post_epochs', type=int, default=10, help='Number of epochs after stopping link prediction.')
parser.add_argument('--mod_freq', type=int, default=1, help='Frequency (epochs) of link prediction.')
parser.add_argument('--num_mods', type=int, default=1, help='Number of changed edges per modification.')