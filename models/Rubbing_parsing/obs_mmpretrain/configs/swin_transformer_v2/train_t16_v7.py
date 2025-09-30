# Only for evaluation
_base_ = [
    # '../_base_/models/swin_transformer_v2/large_384.py',
    # '../_base_/datasets/imagenet_bs64_swin_384.py',
    # '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformerV2',
        arch='tiny',
        img_size=256,
        window_size=[16, 16, 16, 8],
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=88899,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)
# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=88899,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
# set visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(type='RandomSaltPepperNoise', prob=0.5, salt_vs_pepper_ratio_range=(0.1, 0.4), amount_range=(0.002, 0.006)),
                dict(type='RandomErodeDilate',prob=1),
                dict(type='RandomGaussianBlur', p=0.5, min_kernel_size=3, max_kernel_size=15, min_sigma=0.1, max_sigma=1.0),
                dict(type='Resize', scale=(256, 256), keep_ratio=True),
                dict(type='RandomPadding', max_padding=128, mean=[255, 255, 255]),
                dict(type='Resize', scale=(256, 256), keep_ratio=True),
                dict(type='Pad', size=(256, 256), pad_val=dict(img=255)),
                dict(
                    type='RandomAffine_my',
                    max_rotate_degree=5,  # 设置旋转的最大角度范围
                    max_translate_ratio=0.05,  # 禁用平移
                    scaling_ratio_range=(0.9, 1.1),  # 禁用缩放
                    max_shear_degree=0.0,  # 禁用剪切
                    border=(0, 0),  # 无需额外边界
                    border_val=(255, 255, 255),  # 边界填充值
                    bbox_clip_border=True  # 裁剪边界框
                )
            ],
            [
                dict(
                    type='RandomResize',
                    scale=(256, 256),  # 基础目标尺寸
                    ratio_range=(0.5, 1),  # 随机比例范围
                    keep_ratio=True  # 不保持宽高比例
                ),
                # dict(type='Resize', scale=(2048, 256), keep_ratio=True),
                dict(type='Pad', size=(256, 256), pad_val=dict(img=255)),
                dict(type='RandomSaltPepperNoise', prob=0.5, salt_vs_pepper_ratio_range=(0.1, 0.4), amount_range=(0.001, 0.003)),
                dict(type='RandomErodeDilate', prob=0.5),
                dict(type='RandomGaussianBlur', p=0.5, min_kernel_size=3, max_kernel_size=15, min_sigma=0.1,
                     max_sigma=1.0),

            ],
            [
                dict(type='RandomSaltPepperNoise', prob=0.5, salt_vs_pepper_ratio_range=(0.1, 0.4),
                     amount_range=(0.002, 0.006)),
                dict(type='RandomErodeDilate', prob=1),
                dict(type='RandomGaussianBlur', p=0.5, min_kernel_size=3, max_kernel_size=15, min_sigma=0.1,
                     max_sigma=1.0),
                dict(type='Resize', scale=(256, 256), keep_ratio=True),
                dict(type='RandomPadding', max_padding=128, mean=[255, 255, 255]),
                dict(type='Resize', scale=(256, 256), keep_ratio=True),
                dict(type='Pad', size=(256, 256), pad_val=dict(img=255)),
                dict(
                    type='RandomAffine_my',
                    max_rotate_degree=5,  # 设置旋转的最大角度范围
                    max_translate_ratio=0.05,  # 禁用平移
                    scaling_ratio_range=(0.9, 1.1),  # 禁用缩放
                    max_shear_degree=0.0,  # 禁用剪切
                    border=(0, 0),  # 无需额外边界
                    border_val=(255, 255, 255),  # 边界填充值
                    bbox_clip_border=True  # 裁剪边界框
                )
            ],
        ]),
    # dict(type='ColorJitterTransform', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    dict(
        type='PhotoMetricDistortion_my',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(type='PackInputs')
]
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='RandomResizedCrop',
#         scale=384,
#         backend='pillow',
#         interpolation='bicubic'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=256, backend='pillow', interpolation='bicubic'),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='Pad', size=(256, 256), pad_val=dict(img=255)),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    # _delete_=True,
    batch_size=64,
    # persistent_workers=True,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='/data/wpj/jgw_7w/Recognition_v3',
        with_label=True,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    # _delete_=True,
    batch_size=64,
    num_workers=2,
    # persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='/data/wpj/jgw_7w/Recognition_test',
        with_label=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
# load_from='/data/wpj/jgw_7w/mmdetection/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth'
# load_from='/data/wpj/jgw_7w/mmdetection/mmpretrain/work_dirs/train_t16_v6/epoch_400.pth'
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
# test_dataloader = dict(
#     # _delete_=True,
#     batch_size=64,
#     num_workers=2,
#     # persistent_workers=True,
#     dataset=dict(
#         type=dataset_type,
#         data_root='/data/wpj/jgw_7w/mmdetection/',
#         with_label=False,
#         ann_file='/data/wpj/jgw_7w/mmdetection/'+'chinese_v1.txt',
#         pipeline=test_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=False),
# )
test_evaluator = val_evaluator
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5 * 1024 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-6, by_epoch=True, begin=1)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
val_cfg = dict()
test_cfg = dict()
default_hooks = dict(
    # checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3))
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=3,
                    rule='greater'))
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)
