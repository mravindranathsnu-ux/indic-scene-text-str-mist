auto_scale_lr = dict(base_batch_size=16)
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=3, save_best='mist/hmean',
        rule='greater', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False, draw_pred=False, enable=False, interval=1,
        show=False, type='VisualizationHook'))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

mist_images_root = '/home/ravi/mist/Images'
mist_ann_root = '/home/ravi/mist/mmocr_annotations'

mist_textdet_train = dict(
    ann_file=mist_ann_root + '/mist_train_textdet.json',
    data_root=mist_images_root + '/Train',
    data_prefix=dict(img_path=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=4),
    pipeline=None,
    type='OCRDataset')
mist_textdet_test = dict(
    ann_file=mist_ann_root + '/mist_test_textdet.json',
    data_root=mist_images_root + '/Test',
    data_prefix=dict(img_path=''),
    pipeline=None,
    test_mode=True,
    type='OCRDataset')

# Load epoch-40 WEIGHTS ONLY (no optimizer/scheduler resume). This is a
# fresh second fine-tuning phase, not a byte-for-byte continuation of the
# original 40-epoch schedule -- deliberately simpler and more robust than
# trying to resume mmengine's internal scheduler state, which caused two
# failed attempts (stale end -> LR frozen at eta_min; unbuilt scheduler
# object -> AttributeError).
load_from = '/home/ravi/mist_det_work/work_dirs/mist_dbnetpp/epoch_40.pth'
resume = False
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        stage_with_dcn=(False, True, True, True),
        style='pytorch',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        pad_size_divisor=32,
        std=[58.395, 57.12, 57.375],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(
            epsilon_ratio=0.002, text_repr_type='quad',
            type='DBPostprocessor'),
        type='DBHead'),
    neck=dict(
        asf_cfg=dict(attention_type='ScaleChannelSpatial'),
        in_channels=[256, 512, 1024, 2048],
        lateral_channels=256,
        type='FPNC'),
    type='DBNet')

# Lower peak LR than the original run (1e-3): this is continued fine-tuning
# from an already-converged checkpoint, not training from scratch, so a
# smaller "warm restart" LR is standard practice.
optim_wrapper = dict(
    optimizer=dict(lr=0.0002, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')

# Fresh, self-contained 20-epoch decay -- begin=0 within THIS run, not tied
# to the original run's epoch numbering.
param_scheduler = [
    dict(end=20, eta_min=1e-07, power=0.9, type='PolyLR'),
]
randomness = dict(seed=None)
test_cfg = dict(type='TestLoop')

test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(4068, 1024), type='Resize'),
    dict(type='LoadOCRAnnotations', with_bbox=True, with_label=True, with_polygon=True),
    dict(meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor', 'instances'),
         type='PackTextDetInputs'),
]
train_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_bbox=True, with_label=True, with_polygon=True),
    dict(brightness=0.12549019607843137, op='ColorJitter', saturation=0.5, type='TorchVisionWrapper'),
    dict(args=[['Fliplr', 0.5], dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]],
         type='ImgAugWrapper'),
    dict(min_side_ratio=0.1, type='RandomCrop'),
    dict(keep_ratio=True, scale=(640, 640), type='Resize'),
    dict(size=(640, 640), type='Pad'),
    dict(meta_keys=('img_path', 'ori_shape', 'img_shape'), type='PackTextDetInputs'),
]

# Fresh 20-epoch run (not 60) -- epoch numbering restarts at 1 in this
# run's own work_dir, independent of the original run's epoch_40 label.
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(datasets=[mist_textdet_train], pipeline=train_pipeline, type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(datasets=[mist_textdet_test], pipeline=test_pipeline, type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataloader = val_dataloader

val_evaluator = dict(type='HmeanIOUMetric', prefix='mist')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(name='visualizer', type='TextDetLocalVisualizer', vis_backends=vis_backends)
