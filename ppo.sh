for seed in 3; do
    # python train_ipo.py --smooth_cost=True --seed=$seed --name=ipo_01_smooth_seed_$seed
    python train_ppo.py --seed=$seed --name=ppo_01_seed_$seed --env_name=single --num_ped=1 --algo=ppo --batch_size=16 --worker=3 --ppo_epoch=10 --obs_dim=3 --learning_rate=0.001 --eval_rendering=False
done