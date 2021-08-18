def set_wandb_params(wandb, cfg):
    wandb.config.lr = cfg.lr
    wandb.config.batch_size = cfg.batch_size
    wandb.config.image_size = cfg.image_size
    wandb.config.criterion = str(cfg.criterion)
    wandb.config.metric = str(cfg.metric)
    wandb.config.model_name = str(cfg.model_name)
    wandb.config.transform = str(cfg.transform)
    wandb.config.scheduler = cfg.scheduler
    wandb.config.resizemix_aug = str(cfg.resizemix_aug)
    wandb.config.mixup_aug = str(cfg.mixup_aug)
    wandb.config.cutmix_aug = str(cfg.cutmix_aug)
    wandb.config.fmix_aug = str(cfg.fmix_aug)

