auto_scale_lr = dict(base_batch_size=256)
crop_size = (
    128,
    128,
)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=9004,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'Han'
default_hooks = dict(
    checkpoint=dict(interval=10, type='CheckpointHook'),
    logger=dict(interval=200, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=9004,
        topk=(
            1,
            20,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
my_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        128,
        128,
    ), type='Resize'),
    dict(type='PackInputs'),
]
my_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        128,
        128,
    ), type='Resize'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/data/JGW/hanzi2/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                128,
                128,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='Han'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    dl='/home/user/hsguan/mmpretrain/hanzi.txt',
    topk=(
        1,
        20,
    ),
    type='JGWMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/data/JGW/hanzi2/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                128,
                128,
            ), type='Resize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='Han'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/data/JGW/hanzi2/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                128,
                128,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='Han'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        20,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'hanzi50'
