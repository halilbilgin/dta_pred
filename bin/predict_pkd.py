import argparse
from keras.models import load_model
from dta_pred import DataSet
import numpy as np
import pandas as pd
from os import path
import shutil
import json
from dta_pred import argparser, run_experiment, makedirs, logging
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_file',
        type=str,
        help='Name of the CSV file'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Name of the CSV file'
    )

    parser.add_argument(
        '--arguments',
        type=json.loads,
        help='Arguments to train the model'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        help='Trained model file. If this is given, a new model will not be trained'
    )

    FLAGS, _ = parser.parse_known_args()
    args = argparse.Namespace(**FLAGS.arguments)

    if FLAGS.model_file is not None:
        model = load_model(FLAGS.model_file)
    else:
        if not os.path.exists(args.output_path):
            makedirs(args.output_path)
            makedirs(os.path.join(args.log_path))
            makedirs(os.path.join(args.checkpoints_path))

        result = run_experiment({}, args)
        model = load_model(result['checkpoint_files']['Kd'])

    prediction_folder = 'data/prediction'
    makedirs(prediction_folder)

    shutil.copy(FLAGS.input_file, os.path.join(prediction_folder,'test.csv'))
    with open(os.path.join(prediction_folder, 'type.txt'), 'w') as f:
        f.write('Kd')

    data = DataSet(prediction_folder, '', args.max_seq_len, args.max_smi_len,
                       args.protein_format, args.drug_format, args.mol2vec_model_path,
                    args.mol2vec_radius, args.biovec_model_path)

    XD, XT = data.parse_data(with_label=False)

    Y_pred = model.predict([np.asarray(XD), np.asarray(XT)])

    data_csv = pd.read_csv(FLAGS.input_file)
    data_csv.loc[:, 'pKd_[M]_pred'] = Y_pred

    data_csv.to_csv(FLAGS.output_file, index=False)
