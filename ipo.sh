for seed in 3 34 89 233 315 987 1597; do
    # python train_ipo.py --smooth_cost=True --seed=$seed --name=ipo_01_smooth_seed_$seed
    python train_ipo.py --seed=$seed --name=ipo_01_seed_$seed
done