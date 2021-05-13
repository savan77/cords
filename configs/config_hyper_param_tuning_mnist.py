from hyperopt import hp

config = dict(setting= "hyperparamtuning",

# parameter for subset selection
# all settings for subset selection will be fetched from here
subset_config = "configs/config_gradmatchpb-warm_mnist.py",

# parameters for hyper-parameter tuning

# search space for hyper-parameter tuning
space = dict(learning_rate=hp.uniform('lr', 0.0005, 0.01), 
        optimizer= hp.choice('optimizer', ['sgd', 'adam']),
        trn_batch_size= hp.choice('trn_batch_size', [64, 128, 256]),
        num_epochs=hp.choice('num_epochs', [6,7,8,9,10,11,12,13,14,15]),
        kernel1=hp.choice('kernel1', [1,3,5,7]),
        kernel2=hp.choice('kernel2', [1,3,5,7]),
        feat_size1 = hp.choice('feat_size1', [10, 24, 32, 64]),
        feat_size2=hp.choice('feat_size2', [24, 32, 64, 128]),
        fc_deep=hp.choice('fc_deep', [1,2,3,4]),
        fc_width=hp.choice('fc_width', [32, 64, 128, 200]),
        drop1=hp.choice('drop1', [0.25, 0.35, 0.5, 0.7]),
        drop2=hp.choice('drop2', [0.25, 0.35, 0.5, 0.7])),


# tuning algorithm 
search_algo = "TPE",

# number of hyper-parameter set to try
num_evals = 20,

# metric to be optimized, for 'mean_loss' metric mode should be 'min'
metric = "mean_accuracy",
mode = "max",

# scheduler to be used (i.e ASHAScheduler)
# scheduler terminates trials that perform poorly
# learn more here: https://docs.ray.io/en/releases-0.7.1/tune-schedulers.html
scheduler = None,

# where to store logs
log_dir = "/content/drive/MyDrive/RayLogs/",

# resume hyper-parameter tuning from previous log
# specify 'name' (i.e main_2021-03-09_18-33-56) below
resume = False,

# only required if you want to resume from previous checkpoint
# it can also be specified if you don't want to resume
name = None,

# specify resources to be used per trial
# i.e {'gpu':1, 'cpu':2}
resources = {'gpu' : 1},

# if True, trains model on Full dataset with the best parameter selected.
final_train = True

)
