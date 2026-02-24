#!/bin/bash

python src/train.py -m data/train_datasets=[yke_data] \
    data/val_datasets=[yke_data] \
    data/test_datasets=[yke_data] \
    trainer=gpu diffusion.num_timesteps=1000 \
    data.num_workers=8 callbacks=default \
    logger=wandb slurm=null \
    tags=[ykedata_training_2s_1500mHz] \
    trainer.validation_every_n_epochs=1 \
    trainer.test_every_n_epochs=1 \
    trainer.max_epochs=1 \
    train=true test=true \
    evaluation.data_to_extract=['preds','metrics','qualitatives'] \
    evaluation.metrics_to_compute=['multi_match','scan_match','scan_match_no_dur','sequence_score','sequence_score_time','kld','diversity_sequence_score','diversity_sequence_score_time'] \
    diffusion_class=spaced_diffusion \
    callbacks.model_checkpoint.every_n_epochs=1 \
    ckpt_path= checkpoints/scandiff_freeview.pth\