# model settings
model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNeSt',
        depth=200,
        num_stages=4,
        stem_channels=128,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnest200')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        #dcn_on_last_conv=True),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
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
data_root = '../data/rsna18/stage_2_train_images/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadDicomFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
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
    dict(type='LoadDicomFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
    samples_per_gpu=3,
    workers_per_gpu=2,
    ann_file='../data/rsna18/train_bbox_annotations_mmdet.pkl',
    outer_fold=0,
    inner_fold=None,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root),
        classes=['opacity'],
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        classes=['opacity'],
        filter_empty_gt=True),
)
evaluation = dict(interval=1, metric='mAP', save_best='auto', rule='greater')

# optimizer
optimizer = dict(type='AdamW',
    lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

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

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '../experiments/mmdetection/vfn_rsna000'
