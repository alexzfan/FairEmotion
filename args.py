"""
CLI args for the various routines
"""
import argparse

def float_range(mini,maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model on Facial Expression')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=40000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='NLL',
                        choices=('NLL', 'acc'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--model_type',
                        type=str,
                        default = "baseline",
                        choices=("baseline",),
                        help='Model choice for training')
    parser.add_argument('--race_quant_sampling',
                        type=str,
                        default = None,
                        choices=("White","Latino_Hispanic", "Black", "Middle Eastern", "East Asian", "Indian", "Southeast Asian"),
                        help='Race to maintain certain proportion when testing bias quantification')
    parser.add_argument('--race_quant_sampling_prop',
                        type = float_range(0,1),
                        default = None,
                        help='Propertion to maintain for certain race when testing bias quantification')
    parser.add_argument('--shuffle_dataset',
                        type=bool,
                        default=True,
                        help='Shuffle Dataset for Training')
    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument('--train_dir',
                        type=str,
                        default='./data/affectnet/train_set')
    parser.add_argument('--val_dir',
                        type=str,
                        default='./data/affectnet/val_set')
    parser.add_argument("--train_csv", type = str, default="./affectnet_train_filepath_full.csv")
    parser.add_argument("--val_csv", type = str, default="./affectnet_val_filepath_full.csv")
    parser.add_argument('--cafe_train_csv',
                        type=str,
                        default='./cafe_train.csv')
    parser.add_argument('--cafe_val_csv',
                        type=str,
                        default='./cafe_val.csv')
    parser.add_argument('--cafe_test_csv',
                        type=str,
                        default='./cafe_test.csv')




def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=1000,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--load_path_step_bool',
                        type=bool,
                        default=True,
                        help='Whether to use saved step when loading in model')
    parser.add_argument('--dataset',
                        type=str,
                        default="affectnet",
                        help='Dataset to train on')
def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained model on Long Covid Datasets')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')
    parser.add_argument('--model_type',
                        type=str,
                        default = "baseline",
                        choices=("baseline",),
                        help='Model choice for training')
    parser.add_argument('--ensemble_list',
                        type=str,
                        nargs = "+",
                        help='Model best path tars for ensemble',
                        default = [])

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path and not args.ensemble_list:
        raise argparse.ArgumentError('Missing required argument --load_path or --ensemble_list')

    return args