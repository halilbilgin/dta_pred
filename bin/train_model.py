import os

from sacred import Experiment
from sacred.observers import MongoObserver
from datetime import datetime
from dta_pred import argparser, run_experiment, makedirs, logging
from keras import backend as K

if __name__ == "__main__":
    FLAGS = argparser()
    time_str = str(datetime.now().strftime("%x-%X")).replace("/", ".").replace(":", ".")

    FLAGS.output_path = os.path.join(
        FLAGS.output_path, FLAGS.experiment_name + "_" + time_str
    )
    FLAGS.log_path = os.path.join(FLAGS.output_path, "logs")
    FLAGS.checkpoints_path = os.path.join(FLAGS.output_path, "checkpoints")
    ex = Experiment(FLAGS.experiment_name)

    if not os.path.exists(FLAGS.output_path):
        makedirs(FLAGS.output_path)
        makedirs(os.path.join(FLAGS.log_path))
        makedirs(os.path.join(FLAGS.checkpoints_path))

    mongo_conf = FLAGS.mongodb
    if mongo_conf != None:
        mongo_conf = FLAGS.mongodb.split(":")
        ex.observers.append(
            MongoObserver.create(url=":".join(mongo_conf[:-1]), db_name=mongo_conf[-1])
        )

    logging(str(FLAGS), FLAGS.log_path)

    ex.main(run_experiment)
    cfg = vars(FLAGS)
    cfg["FLAGS"] = FLAGS
    ex.add_config(cfg)

    r = ex.run()


# KD-GIP and KP-GS-domain
# kernels, followed closely by KD-GIP and KP-SW+ k
