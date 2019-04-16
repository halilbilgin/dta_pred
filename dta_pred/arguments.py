import argparse
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='experiment',
        help='Name of the experiment.'
    )
    parser.add_argument(
        '--smi_model',
        type=str,
        help='Model for encoding SMILES. i.e, simple_cnn, inception'
    )

    parser.add_argument(
        '--seq_model',
        type=str,
        help="Model for encoding Sequences. i.e, simple_cnn, inception"
    )

    parser.add_argument(
        '--smi_window_length',
        type=int,
        help='smiles filter length'
    )
    parser.add_argument(
        '--seq_window_length',
        type=int,
        help='motif filter length'
    )
    parser.add_argument(
        '--n_cnn_layers',
        type=int,
        default=3,
        help='Number of CNN layers'
    )

    parser.add_argument(
        '--num_windows',
        type=int,
        help='Number of filters in CNNs'
    )
    parser.add_argument(
        '--n_fc_neurons',
        type=int,
        default=1024,
        help='Number of neurons in fully connected hidden layers.'
    )

    parser.add_argument(
        '--n_fc_layers',
        type=int,
        default=1,
        help='Number of Fully Connected layers in interaction modeling stage'
    )

    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
        help='Length of input sequences.'
    )
    parser.add_argument(
        '--max_smi_len',
        type=int,
        default=None,
        help='Length of input sequences.'
    )
    parser.add_argument(
        '--protein_format',
        type=str,
        default='sequence',
        help='Format of the proteins, i.e. sequence, pssm'
    )

    # for learning
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=100,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--apply_bn',
        type=int,
        default=0,
        help='Whether apply batch normalization in fully connected layers or not'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='Optimization method to use during training, https://keras.io/optimizers/'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.4,
        help='Dropout level.'
    )
    parser.add_argument(
            '--cross_validation',
            type=bool,
            default=False,
            help='Dropout level.'
        )

    parser.add_argument(
        '--binary_th',
        type=float,
        default=7.0,
        help='Threshold to split data into binary classes'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='../../data/',
        help='Directory for input data.'
    )
    parser.add_argument(
        '--datasets_included',
        type=str,
        nargs='+',
        default=['davis', 'dtc_KD', 'dtc_IC50', 'dtc_KI', 'kiba'],
        help='Directory for input data.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='output/',
        help='Path to use for output.'
    )
    parser.add_argument(
        '--mongodb',
        type=str,
        default=None,
        help='MongoDB configuration'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for ensuring reproducibility'
    )
    parser.add_argument(
        '--drug_format',
        type=str,
        default='labeled_smiles',
        help='Format of the drug, i.e. labeled_smiles, mol2vec'
    )
    parser.add_argument(
        '--mol2vec_model_path',
        type=str,
        default="trained_models/mol2vec/model_300dim.pkl",
        help='Mol2vec model path. Used if drug_format="mol2vec"'
    )
    parser.add_argument(
        '--mol2vec_radius',
        type=int,
        default=1,
        help='Radius utilized for Morgan algorithm. Used if drug_format="mol2vec"'
    )
    parser.add_argument(
        '--mol2vec_output_dim',
        type=int,
        default=300,
        help='Output dimension of mol2vec model. Used if drug_format="mol2vec"'
    )
    parser.add_argument(
        '--biovec_model_path',
        type=str,
        default='trained_models/biovec/swissprot-reviewed.model',
        help='Output dimension of biovec model. Used if protein_format="biovec"'
    )
    parser.add_argument(
        '--biovec_output_dim',
        type=int,
        default=300,
        help='Output dimension of biovec model. Used if protein_format="biovec"'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='mean_squared_error',
        help='Loss function. It can take mean_squared_error and crossentropy_mse_combined'
    )
    parser.add_argument(
        '--l2_regularizer_fc',
        type=float,
        default=0,
        help='L2 regularization parameter for fully connected layers'
    )

    FLAGS, unparsed = parser.parse_known_args()

    # check validity
    #assert( len(FLAGS.window_lengths) == len(FLAGS.num_windows) )

    return FLAGS

def logging(msg, log_path):
    fpath = os.path.join( log_path, "log.txt" )
    with open( fpath, "a" ) as fw:
        fw.write("%s\n" % msg)
        print(msg)

    #print(msg)
