_base_ = [
    '../_base_/models/swin_transformer_v2/base_256.py',
    '../_base_/datasets/imagenet_bs64_swin_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'JGW'
crop_size = (256, 256)
data_preprocessor = dict(
    num_classes=13717,
    # to_rgb=False,
)

model = dict(
    head=dict(num_classes=13717,
              topk=(1, 20),
              )
)

# train_pipeline = [
#     dict(type='LoadImageFromFile', imdecode_backend='pillow'),
#     dict(type='RandomResizedCrop', scale=224, backend='pillow'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]
my_train_pipeline = [
    # dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

my_test_pipeline = [
    # dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/data/data_l/', type=dataset_type, pipeline=my_train_pipeline),
    batch_size=64,
)

val_dataloader = dict(
    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/data/data_l/', type=dataset_type
                 , pipeline=my_test_pipeline),
    batch_size=64,
)

test_dataloader = val_dataloader
# test_dataloader = dict(
#    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/k/', type=dataset_type, split='test', pipeline=test_pipeline),
# )

# auto_scale_lr = dict(base_batch_size=32)

val_evaluator = dict(type='Accuracy', topk=(1, 20))

# If you want standard test, please manually configure the test dataset
test_evaluator = dict(type='JGWMetric', topk=(1, 20), dl='/data/JGW/hsguan/mmpretrain/swin_v2_datal.txt')

train_cfg = dict(max_epochs=300)  # 训练 300 个 epoch，每 10 个 epoch 评估一次

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * 64 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),)

# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=0.25,
#         by_epoch=True,
#         begin=0,
#         end=5,
#         # update by iter
#         convert_to_iter_based=True,
#     ),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=295,
#         by_epoch=True,
#         begin=10,
#         end=300,
#     )
# ]

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]

auto_scale_lr = dict(base_batch_size=64)

default_hooks = dict(

    logger=dict(interval=200),

    checkpoint=dict(interval=10),

)