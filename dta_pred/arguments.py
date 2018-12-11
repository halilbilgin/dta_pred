import argparse
import os

def argparser():
  parser = argparse.ArgumentParser()

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
      '--num_windows',
      type=int,
      help='Number of filters in CNNs'
  )
  parser.add_argument(
      '--n_neurons_fc',
      type=int,
      default=1024,
      help='Number of neurons in fully connected hidden layers.'
  )
  parser.add_argument(
      '--n_cnn_layers',
      type=int,
      default=3,
      help='Number of CNN layers'
  )
  parser.add_argument(
      '--resampling',
      type=str,
      default='over',
      help='Resampling method for train set. It can take over, under and None'
  )
  parser.add_argument(
      '--pool_size',
      type=int,
      default=2,
      help='Pool size to use if pooling_type is MaxPooling'
  )
  parser.add_argument(
      '--pooling_type',
      type=str,
      default="GlobalMaxPooling",
      help='Pool type. i.e. MaxPooling, GlobalAveragePooling'
  )
  parser.add_argument(
      '--n_fc_layers',
      type=int,
      default=1,
      help='Number of Fully Connected layers'
  )
  parser.add_argument(
      '--num_classes',
      type=int,
      default=0,
      help='Number of classes (families).'
  )
  parser.add_argument(
      '--max_seq_len',
      type=int,
      default=0,
      help='Length of input sequences.'
  )
  parser.add_argument(
      '--max_smi_len',
      type=int,
      default=0,
      help='Length of input sequences.'
  )
  parser.add_argument(
      '--protein_format',
      type=str,
      default='sequence',
      help='Format of the proteins, i.e. sequence, pssm'
  )
  parser.add_argument(
      '--drug_format',
      type=str,
      default='SMILES',
      help='Format of the drug, i.e. SMILES, VAE_code'
  )
  parser.add_argument(
      '--drug_vae_code_len',
      type=int,
      help='Length of the feature representation vector, only used if drug_format="VAE_code"'
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
      '--dataset_path',
      type=str,
      default='../../data/davis',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--dtc_path',
      type=str,
      default='../../data/dtc',
      help='Directory for DTC data.'
  )
  parser.add_argument(
      '--dropout',
      type=float,
      default=0.4,
      help='Dropout level.'
  )
  parser.add_argument(
      '--binary_th',
      type=float,
      default=0.0,
      help='Threshold to split data into binary classes'
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp',
      help='Directory for log data.'
  )
  parser.add_argument(
      '--mongodb',
      type=str,
      default=None,
      help='MongoDB configuration'
  )

  FLAGS, unparsed = parser.parse_known_args()

  # check validity
  #assert( len(FLAGS.window_lengths) == len(FLAGS.num_windows) )

  return FLAGS




def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  #print(msg)
