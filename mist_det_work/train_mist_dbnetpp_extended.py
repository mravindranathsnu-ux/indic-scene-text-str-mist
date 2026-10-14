from mmengine.config import Config
from mmengine.runner import Runner
from mmocr.utils import register_all_modules

register_all_modules()
cfg = Config.fromfile('/home/ravi/mist_det_work/mist_dbnetpp_config_extended.py')

# Separate work_dir so this run's checkpoints (epoch_1.pth, epoch_5.pth, ...
# up to epoch_20.pth, best_mist_hmean_epoch_N.pth) never touch or overwrite
# the original run's files in work_dirs/mist_dbnetpp/ -- that original best
# checkpoint (68.08%/61.04% DetEval result) stays intact as a fallback no
# matter what this run does.
cfg.work_dir = '/home/ravi/mist_det_work/work_dirs/mist_dbnetpp_ext'

runner = Runner.from_cfg(cfg)
runner.train()
