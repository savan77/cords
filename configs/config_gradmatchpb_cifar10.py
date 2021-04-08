# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="cifar100",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='ResNet50',
                         type='pre-defined',
                         numclasses=100),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.01,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_strategy=dict(type="GradMatchPB",
                                fraction=0.1,
                                select_every=20,
                                lam=0),

              train_args=dict(num_epochs=300,
                              device="cuda",
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc"],
                              return_args=[]
                              )
              )
