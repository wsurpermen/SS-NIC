# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import run_lib
import torch
import numpy as np
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import train_classify
from models.utils import after_defined

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "fine_tune", 'eval',"train_classify"], "Run training")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_float("noise", 0.05, "noise ratio --control the ratio of noise content")
flags.DEFINE_bool("cond_y", False, "if train score(x|y)")
flags.DEFINE_bool("xscore_label", False, "if train score(x, y)")
flags.mark_flags_as_required(["workdir", "config", "mode", "noise", 'xscore_label', 'cond_y'])


def main(argv):
    randomSeed = 2021
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(randomSeed)
    FLAGS.config.training.noise = FLAGS.noise

    if FLAGS.mode == "train":
        # Create the working directoryz
        os.makedirs(FLAGS.workdir, exist_ok=True)
        after_defined(FLAGS,FLAGS.config)

        os.makedirs(os.path.join(FLAGS.workdir, 'samples'), exist_ok=True)
        # Set logger so that it outputs to both console and file
        gfile_stream = open(os.path.join(FLAGS.workdir, 'train.txt'), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')

        run_lib.train(FLAGS.config, FLAGS.workdir)

    elif FLAGS.mode == "train_classify":
        # Create the working directory
        os.makedirs(FLAGS.workdir, exist_ok=True)
        # Set logger so that it outputs to both console and file
        gfile_stream = open(os.path.join(FLAGS.workdir, 'train.txt'), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        FLAGS.config.training.label = True
        train_classify.train(FLAGS.config, FLAGS.workdir)

    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
