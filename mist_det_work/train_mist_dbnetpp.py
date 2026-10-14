from mmengine.config import Config
from mmengine.runner import Runner
from mmocr.utils import register_all_modules

register_all_modules()
cfg = Config.fromfile('/home/ravi/mist_det_work/mist_dbnetpp_config.py')
cfg.work_dir = '/home/ravi/mist_det_work/work_dirs/mist_dbnetpp'
runner = Runner.from_cfg(cfg)
runner.train()
