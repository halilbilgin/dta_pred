import os

from sacred import Experiment
from sacred.observers import MongoObserver

from dta_pred import argparser, run_experiment

if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = os.path.join(FLAGS.log_dir, FLAGS.experiment_name)

    ex = Experiment(FLAGS.experiment_name)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    mongo_conf = FLAGS.mongodb
    if mongo_conf != None:
        mongo_conf = FLAGS.mongodb.split(':')
        ex.observers.append(MongoObserver.create(url=':'.join(mongo_conf[:2]), db_name=mongo_conf[2]))

    ex.main(run_experiment)
    cfg = vars(FLAGS)
    cfg['FLAGS'] = FLAGS
    ex.add_config(cfg)

    r = ex.run()

