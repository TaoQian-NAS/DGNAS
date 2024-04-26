import argparse


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--batch_size', type=int, default='0')

    # setting for transformation
    parser.add_argument('--skip_connections', type=str, default='No')
    parser.add_argument('--dim_hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--activation_function', type=str, default='relu6')
    parser.add_argument('--num_layers', type=int, default=12)

    # setting for propagation
    parser.add_argument('--graph_normalization', type=str, default='MeanNorm')
    parser.add_argument('--propagation_layers', type=int, default=1)
    parser.add_argument('--aggregator_type', type=str, default='mean')

    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0008,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (1 - keep probability).')

    # setting for the Particle swarm optimization
    parser.add_argument("--c1", type=float, default=1.8,
                        help='pso parameter c1 ')
    parser.add_argument("--c2", type=float, default=1.8,
                        help='pso parameter c1 ')
    parser.add_argument("--W", type=float, default=0.729,
                        help='pso parameter W ')
    parser.add_argument('--particle_dim', type=int, default=10,
                        help='the dimension of each particle')
    parser.add_argument('--particle_num', type=int, default=25,
                        help='the size of particle swarm')
    parser.add_argument('--iterations', type=int, default=50,
                        help='the maximum number of iterations')

    args = parser.parse_args()

    return args


def reset_pso_parameters(args, transformation_selection, propagation_selection, param_selection):
    args.skip_connections = transformation_selection[0]
    args.dim_hidden = transformation_selection[1]
    args.activation_function = transformation_selection[2]
    args.num_layers = transformation_selection[3]

    args.graph_normalization = propagation_selection[0]
    args.propagation_layers = propagation_selection[1]
    args.aggregator_type = propagation_selection[2]

    args.dropout = param_selection[0]
    args.lr = param_selection[1]
    args.weight_decay = param_selection[2]


