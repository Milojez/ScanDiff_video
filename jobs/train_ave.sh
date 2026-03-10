#!/bin/bash
# evaluation.data_to_extract=['preds','metrics','qualitatives'] \
# evaluation.metrics_to_compute=['multi_match','scan_match','scan_match_no_dur','sequence_score','sequence_score_time','kld','diversity_sequence_score','diversity_sequence_score_time']
#what is callbacks.model_checkpoint.every_n_epochs=10 - savesa checpoint every n epochs
# ckpt_path="./checkpoints/scandiff_freeview.pth" \
python src/train.py -m data/train_datasets=[ykedata] \
    data/val_datasets=[ykedata] \
    data/test_datasets=[ykedata] \
    trainer=gpu diffusion.num_timesteps=50 \
    data.num_workers=8 callbacks=default \
    logger=wandb ~slurm \
    tags=[ykedata_training_2s_1500mHz] \
    trainer.validation_every_n_epochs=9999 \
    trainer.test_every_n_epochs=9999 \
    trainer.max_epochs=40 \
    train=true test=true \
    evaluation.data_to_extract=['preds','metrics','qualitatives'] \
    evaluation.metrics_to_compute=['multi_match','scan_match','scan_match_no_dur','kld'] \
    diffusion_class=spaced_diffusion \
    callbacks.model_checkpoint.every_n_epochs=100 \
    ckpt_path=null \

