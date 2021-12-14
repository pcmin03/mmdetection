dataset_type = 'CocoDataset'
data_root = '/data/cmpark/sartorius/'
img_norm_cfg = dict(
    mean=[128, 128, 128], std=[11.58, 11.58, 11.58], to_rgb=True)

albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625,
         scale_limit=0.15, rotate_limit=15, p=0.4),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2,
         contrast_limit=0.2, p=0.5),
    dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
    dict(type='CLAHE', p=0.5)
    # dict(
    #     type="OneOf",
    #     transforms=[
    #         dict(type="GaussianBlur", p=1.0, blur_limit=7),
    #         dict(type="MedianBlur", p=1.0, blur_limit=7),
    #     ],
    #     p=0.4,
    # ),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=[(1333, 800), (1690, 960)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    
    # dict(type="Mosaic"),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes', gt_masks='masks'),
        update_pad_shape=False,
        skip_img_without_anno=True),

    # dict(type="MixUp", min_bbox_size=5, pad_val=img_norm_cfg["mean"][0]),

    dict(
        type='Normalize',
        mean=[128, 128, 128],
        std=[11.58, 11.58, 11.58],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'), 
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_masks', 'gt_labels'])

]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(704, 520), (704, 520)],
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'label_info/binary_cocotype/train_0_fold.json',
        img_prefix=data_root + '',
        classes = 'cell',
        pipeline=train_pipeline),
        
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'label_info/binary_cocotype/valid_0_fold.json',
        img_prefix=data_root + '',
        classes = 'cell',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'label_info/binary_cocotype/valid_0_fold.json',
        img_prefix=data_root + '',
        classes = 'cell',
        pipeline=test_pipeline))
evaluation = dict(metric='segm', interval=1, by_epoch=True)
