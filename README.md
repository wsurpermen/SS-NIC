# SS-NIC
Leveraging the divergence between label-induced and feature distributions to identify label noise


# to train diffusion model with out label
python3 main.py --config configs/iris.py --mode train --workdir iris_woy_xscore
# to train classifier 
python3 main.py --config configs/iris.py --mode train_classify --workdir iris_classify
# to train diffusion model with label
python3 main.py --config configs/iris.py --cond_y True --mode train --workdir iris_condy_xscore
# to identify the noise and get the best result
python3 bayesian_tune.py --name iris
# the baseline method 
python3 another.py --name iris --noise 0.4
