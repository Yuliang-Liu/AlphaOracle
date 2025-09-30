_base_ = [
    'co_dino_5scale_r50_lsj_8xb2_1x_JGW.py',
]

num_dec_layer = 6
loss_lambda = 2.0
num_classes = 1

max_epochs = 50
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

data_root = '/data/JGW/mmdet/'
# dataset_type = 'JgwCocoDataset'
dataset_type = 'JgwCocoDetDataset'
sentence = True

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        sentence=sentence,
        # ann_file='annotations/oracle_train_coco.json',
        ann_file='annotations/oracle_train_coco_detection.json',
        data_prefix=dict(img='moben/')
    )
)

val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        sentence=sentence,
        # ann_file='annotations/oracle_test_coco.json',
        ann_file='annotations/oracle_test_coco_detection.json',
        data_prefix=dict(img='moben/')
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoJgwMetric',
    sentence=sentence,
    # ann_file=data_root + 'annotations/oracle_test_coco.json'
    ann_file=data_root + 'annotations/oracle_test_coco_detection.json',
    # iou_thrs=[0.5]
)
test_evaluator = val_evaluator

# optim_wrapper = dict(accumulative_counts=2)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        type='CheckpointHook',
        save_best='auto'
    )
)

vis_backends = [dict(type='LocalVisBackend'),
                # dict(type='TensorboardVisBackend'),
                # dict(type='WandbVisBackend')
                ]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

auto_scale_lr = dict(base_batch_size=16)
