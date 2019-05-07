import argparse
from keras.models import load_model
from dta_pred import DataSet
import numpy as np
import pandas as pd
from os import path

import json
from dta_pred import argparser, run_experiment, makedirs, logging
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_filename',
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



    data = DataSet('data/leaderboards', 'round2/', args.max_seq_len, args.max_smi_len,
                       args.protein_format, args.drug_format, args.mol2vec_model_path,
                    args.mol2vec_radius, args.biovec_model_path)

    XD, XT = data.parse_data(with_label=False)

    Y_pred = model.predict([np.asarray(XD), np.asarray(XT)])

    data_csv = pd.read_csv(path.join('data/leaderboards/round2/template.csv'))
    data_csv.loc[:, 'pKd_[M]_pred'] = Y_pred

    data_csv.to_csv(FLAGS.output_filename, index=False)
