dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '../experiments/mmdetection/swin_rsna000/latest.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = '../experiments/mmdetection/swin000'
fp16 = dict(loss_scale='dynamic')

# model settings
model = dict(
    type='VFNet',
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
        use_checkpoint=False,),
        #pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth'),
        #pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth'),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
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
data_root = '../data/covid/train_pngs/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(512, 512),
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
        img_scale=(512, 512),
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
        classes=['opacity']),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        classes=['opacity'],
        filter_empty_gt=False),
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
    step=[4, 6])
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