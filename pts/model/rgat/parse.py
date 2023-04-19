import argparse

parser = argparse.ArgumentParser()

# TS Data
parser.add_argument('--dataset', type=str, default="exchange_rate", help='Dataset.')
parser.add_argument('--target_dim', type=int, default=0, help='multivariate dimension.')

# Graph
parser.add_argument('--graph_constr', type=str, default="full", help='Graph construction method.')
parser.add_argument('--graph_density', type=float, default=0.2, help='Density of graph.')

# Model
parser.add_argument('--lin_hidden', type=int, default=32, help='Number of neurons in decoder.')
parser.add_argument('--att_hidden', type=int, default=0, help='Number of neurons in attention mechanism.')
parser.add_argument('--embedding_dim', type=int, default=4, help='Embedding dimension.')
parser.add_argument('--num_att_heads', type=int, default=8, help='Number of attention heads.')
parser.add_argument('--activation', type=str, default='nn.Tanh()')
parser.add_argument('--activation_att', type=str, default='nn.LeakyReLU(negative_slope=0.2)')
parser.add_argument('--dropout', type=float, default=0., help='Dropout probability (input).')
parser.add_argument('--dropout_att', type=float, default=0., help='Dropout probability (attention).')

# Training
parser.add_argument('--seed_torch', type=int, default=0, help='torch seed.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs.')
parser.add_argument('--accelerator', type=str, default='cuda', choices=['cuda', 'cpu'], help='Use or gpu or cpu.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_batches_per_epoch', type=int, default=100, help='Number of batches per epochs.')
parser.add_argument('--scaling', type=str, default='True', help='Use mean scaling.')
parser.add_argument('--link_prediction', action='store_true', help='Link prediction.')
parser.add_argument('--pre_epochs', type=int, default=0, help='Number of epochs before starting link prediction.')
parser.add_argument('--post_epochs', type=int, default=0, help='Number of epochs after stopping link prediction.')
parser.add_argument('--mod_freq', type=int, default=1, help='Interval (epochs) of link prediction.')
parser.add_argument('--num_mods', type=int, default=1, help='Number of changed edges per modification.')
parser.add_argument('--project_name', type=str, default='pts-rgat', help='Wandb project name.')