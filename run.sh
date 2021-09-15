#!/bin/bash
export PYTHONWARNINGS="ignore"

python /cephfs/jianyu/GANPSMNetS/gantrain.py \
    --datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --trainlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_train.txt \
    --test_datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --testlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_val.txt \
    --test_sim_datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training \
    --sim_testlist /cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training_lists/all.txt \
    --test_real_datapath /cephfs/datasets/iccv_pnp/real_data_v9/ \
    --real_testlist /cephfs/jianyu/newTrain.txt \
    --depthpath /cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training \
    --epochs 1 \
    --lrepochs "200:10" \
    --crop_width 512  \
    --crop_height 256 \
    --test_crop_width 960  \
    --test_crop_height 540 \
    --using_ns \
    --ns_size 3 \
    --cmodel stackhourglass \
    --logdir "/cephfs/jianyu/eval/psm_gan_train_1ly"  \
    --cbatch_size 1 \
    --test_batch_size 1 \
    --summary_freq 500 \
    --test_summary_freq 500 \
    --loadmodel "/cephfs/jianyu/eval/psm_eval/checkpoint_0.tar"


