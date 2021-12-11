dataset_type = 'CocoDataset'
data_root = '/data/cmpark/sartorius/'
img_norm_cfg = dict(
    mean=[127.96497969, 127.96497969, 127.96497969], std=[13.68662335, 13.68662335, 13.68662335], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'label_info/cocotype/train_0_fold.json',
        img_prefix=data_root + 'train/',
        classes = ('shsy5y','cort','astro') ,
        pipeline=train_pipeline),
        
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'label_info/cocotype/valid_0_fold.json',
        img_prefix=data_root + 'train/',
        classes = ('shsy5y','cort','astro') ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'label_info/cocotype/valid_0_fold.json',
        img_prefix=data_root + 'train/',
        classes = ('shsy5y','cort','astro') ,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
