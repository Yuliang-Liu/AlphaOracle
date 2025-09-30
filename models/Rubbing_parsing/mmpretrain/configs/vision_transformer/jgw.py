_base_ = './vit-large-p16_64xb64_in1k.py'

dataset_type = 'JGW'

data_preprocessor = dict(
    num_classes=9228,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=False,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

model = dict(
    head=dict(num_classes=9228,
              topk=(1, 20),
              )
)

train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

#train_pipeline = [
#    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
#    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
#    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#    dict(type='PackInputs'),
#]
#
#test_pipeline = [
#    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
#    dict(type='ResizeEdge', scale=256, edge='short'),
#    dict(type='CenterCrop', crop_size=224),
#    dict(type='PackInputs'),
#]



train_dataloader = dict(
    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/data/JGWN2/', type=dataset_type, pipeline=train_pipeline),
)

val_dataloader = dict(
    dataset=dict(data_root='/data/JGW/hsguan/mmpretrain/data/JGWN2/', type=dataset_type, pipeline=test_pipeline),
)

test_dataloader = val_dataloader

#auto_scale_lr = dict(base_batch_size=32)

val_evaluator = dict(type='Accuracy', topk=(1, 20))

# If you want standard test, please manually configure the test dataset
test_evaluator = dict(type='JGWMetric', topk=(1, 20), dl='/data/JGW/hsguan/mmpretrain/vit.txt')

#train_cfg = dict(max_epochs=300)  # 训练 300 个 epoch，每 10 个 epoch 评估一次


default_hooks = dict(
    logger=dict(interval=50),
    checkpoint=dict(interval=10),
)


#optim_wrapper = dict(
#    optimizer=dict(type='AdamW', lr=0.1, weight_decay=0.3),
#)