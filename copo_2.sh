for seed in 3 34 89 233 315 987 1597; do
    # python train_ipo.py --smooth_cost=True --seed=$seed --name=ipo_01_smooth_seed_$seed
    python train_copo_2.py --seed=$seed --name=copo_new_seed_$seed --env_name=easy --worker=4
done