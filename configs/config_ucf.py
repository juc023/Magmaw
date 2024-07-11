from configs.config import cfg


dataset_type = 'RawframeDataset'
data_dir = cfg.video_data_dir
data_root = data_dir + 'rawframes'
split = 1
ann_file_train = data_dir + f'ucf101_train_split_{split}_rawframes.txt'
ann_file_val = data_dir + f'ucf101_val_split_{split}_rawframes.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)
#img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', 
         clip_len=16, 
         frame_interval=1, 
         num_clips=1,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),
    dict(type='CenterCrop', crop_size=128),
    dict(type='Resize', scale=(128, -1)),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),
    dict(type='CenterCrop', crop_size=128),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=30, # with num_clip=1, this means batch_size = 15
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline))
