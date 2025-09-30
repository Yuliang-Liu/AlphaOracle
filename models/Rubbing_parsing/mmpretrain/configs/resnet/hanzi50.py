_base_ = './resnet50_8xb32_in1k.py'

dataset_type = 'Han'
crop_size = (128, 128)
data_preprocessor = dict(
    num_classes=9004,
    # to_rgb=False,
)

model = dict(
    head=dict(num_classes=9004,
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
    num_workers=4,
    dataset=dict(data_root='/data/JGW/hanzi2/', type=dataset_type, pipeline=my_train_pipeline),
    batch_size=64,
)

val_dataloader = dict(
    num_workers=4,
    dataset=dict(data_root='/data/JGW/hanzi2/', type=dataset_type
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
test_evaluator = dict(type='JGWMetric', topk=(1, 20), dl='/home/user/hsguan/mmpretrain/hanzi.txt')

train_cfg = dict(max_epochs=100)  # 训练 300 个 epoch，每 10 个 epoch 评估一次

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1))

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

# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[120, 240, 360], gamma=0.1)

default_hooks = dict(

    logger=dict(interval=200),

    checkpoint=dict(interval=10),

)
