dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '../experiments/mmdetection/swin_rsna002/latest.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = '../experiments/mmdetection/swin005/'
fp16 = dict(loss_scale='dynamic')

# model settings
model = dict(
    type='RepPointsDetector',
    backbone=dict(
        type='SwinTransformer',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        pretrained=None),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='moment'),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomGamma'),
            dict(type='RandomContrast'),
            dict(type='RandomBrightness'),
            dict(type='ShiftScaleRotate',
                 shift_limit=0.10,
                 scale_limit=0,
                 rotate_limit=0),
            dict(type='ShiftScaleRotate',
                 shift_limit=0,
                 scale_limit=0.15,
                 rotate_limit=0),
            dict(type='ShiftScaleRotate',
                 shift_limit=0,
                 scale_limit=0,
                 rotate_limit=30),
            dict(type='GaussianBlur'),
            dict(type='IAAAdditiveGaussianNoise')
        ], 
        p=1.0)
] * 3

# data setting
dataset_type = 'CXRDataset'
data_root = '../data/covid/train_pngs/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(512, 512), (768, 768)],
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512), (640, 640), (768, 768)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
infer_pipeline = [
    dict(type='LoadDicomFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512), (640, 640), (768, 768)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    ann_file='../data/covid/train_bbox_annotations_mmdet.pkl',
    outer_fold=0,
    inner_fold=None,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        classes=['opacity'],
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        classes=['opacity'],
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        pipeline=infer_pipeline,
        data_root='',
        filter_empty_gt=False)
)
evaluation = dict(interval=1, metric='mAP', save_best='auto', rule='greater')

# optimizer
optimizer = dict(type='AdamW',
    lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-4)
runner = dict(type='EpochBasedRunner', max_epochs=8)

# runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]