# optimizer
optimizer = dict(type='SGD', lr=0.0025,  momentum=0.9,weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
lr_config = dict(
    policy='CosineAnnealing', 
    by_epoch=False,
    warmup='linear', 
    warmup_iters=125, 
    warmup_ratio=0.001,
    min_lr=1e-07)

runner = dict(type='EpochBasedRunner', max_epochs=50)
