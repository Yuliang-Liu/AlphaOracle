from mmengine import Config

cfg = Config.fromfile('./configs/codino/co_dino_5scale_r50_1x_JGW.py')
print(cfg.model.roi_head)
