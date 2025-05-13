import os
import random
import time

import debugpy
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from poser.dataset import BaseDataModule
from poser.model import OptionalLookAtModel
from utils.path_manager import PathManager

torch.autograd.set_detect_anomaly(True)


def run(version_name, batch_size, max_epochs, num_workers, persistent_workers):
    if torch.cuda.is_available():
        print("\033[92m", "Using CUDA.", "\033[0m", sep="")
    else:
        print("\033[93m", "CUDA not available!", "\033[0m", sep="")
        print("\033[91m", "Installing dependencies in blender breaks this Conda environment.", "\033[0m", sep="")
        exit(0)

    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    rnd_seed = 42
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    path_manager = PathManager()
    logger = TensorBoardLogger(save_dir=f"{path_manager.models_path}/{version_name}")
    model_checkpoint = ModelCheckpoint(dirpath=logger.log_dir + '/checkpoints', save_top_k=1, monitor=None, mode="min")

    data_module = BaseDataModule(batch_size, num_workers, persistent_workers)
    model = OptionalLookAtModel()
    trainer = pl.Trainer(logger=[logger], callbacks=[model_checkpoint], accelerator='auto', devices=1, max_epochs=max_epochs, gradient_clip_val=1.0)
    trainer.fit(model, datamodule=data_module)

    end_time = time.time()
    print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    total_time = end_time - start_time
    total_time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
    if os.environ.get('PL_WORKER_ID') is None:
        try:
            debugpy.listen(("0.0.0.0", 5678))
            print("\033[93m", "Debugpy is listening for connections...", "\033[0m", sep="")
            # debugpy.wait_for_client()
            # print("Debugger attached!")
        except RuntimeError as e:
            print("\033[93m", f"Debugpy setup failed: {e}", "\033[0m", sep="")

    run(
        version_name="2000_epochs",
        # batch_size=1024,
        batch_size=512,
        max_epochs=2000,
        # max_epochs=1000,
        # max_epochs=100,
        # max_epochs=2,

        num_workers=4,
        persistent_workers=True,
        # num_workers=0,
        # persistent_workers=False,
    )

# python -m poser.trainer
# python -m poser.trainer > training_output.log 2>&1

# tensorboard --logdir=poser\models
# http://localhost:6006/

# pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
