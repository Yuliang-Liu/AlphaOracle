_base_ = './resnet101_8xb32_in1k.py'

dataset_type = 'JGW'

data_preprocessor = dict(
    num_classes=997,
)

model = dict(
    head=dict(num_classes=997,
              topk=(1, 20),
              )
)

train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/data/JGW/', type=dataset_type, pipeline=train_pipeline),
)

val_dataloader = dict(
    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/data/JGW/', type=dataset_type, pipeline=test_pipeline),
)

#test_dataloader = val_dataloader
test_dataloader = dict(
    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/k/', type=dataset_type, split='test', pipeline=test_pipeline),
)

#auto_scale_lr = dict(base_batch_size=32)

val_evaluator = dict(type='Accuracy', topk=(1, 20))

# If you want standard test, please manually configure the test dataset
test_evaluator = dict(type='JGWMetric', topk=(1, 20), dl='/data/JGW/hsguan/mmpretrain/k.txt')

train_cfg = dict(max_epochs=300)  # 训练 300 个 epoch，每 10 个 epoch 评估一次
param_scheduler = dict(milestones=[100, 200, 300], gamma=0.05)

default_hooks = dict(

    logger=dict(interval=5),

    checkpoint=dict(interval=10),

)