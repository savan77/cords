# -*- coding: utf-8 -*-
"""Modular_Training_Warmstart_Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JTRI_p26DY8-cTq16YPGPcyYCUrlKiM_
"""

import time
import datetime
import copy
import numpy as np
import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from cords.selectionstrategies.supervisedlearning import OMPGradMatchStrategy, GLISTERStrategy, RandomStrategy, \
    CRAIGStrategy
from cords.utils.models import ResNet18, MnistNet, ResNet164
from cords.utils.custom_dataset import load_dataset_custom
from torch.utils.data import Subset
from math import floor
from hyperopt import hp, tpe, fmin, Trials, STATUS_FAIL, STATUS_OK
import uuid
import pickle


#global space var
space = {'lr': hp.uniform('lr', 0.01, 0.0007), 
        'optimizer': hp.choice('optimizer', ['Adam', 'SGD']),
        'trn_batch_size': hp.choice('trn_batch_size', [20, 32, 64])
        }
datadir = '../../data'
data_name = 'cifar10'
fraction = float(0.1)
num_epochs = int(100)
select_every = int(20)
feature = 'dss'
num_runs = 1  # number of random runs
learning_rate = 0.01
model_name = 'ResNet18'
device = "cuda" if torch.cuda.is_available() else "cpu"
strategy = 'GradMatch'

def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss


def create_model(name, num_cls, device):
    if name == 'ResNet18':
        model = ResNet18(num_cls)
    elif name == 'MnistNet':
        model = MnistNet()
    elif name == 'ResNet164':
        model = ResNet164(num_cls)
    model = model.to(device)
    return model


"""#Loss Type, Optimizer and Learning Rate Scheduler"""


def loss_function():
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    return criterion, criterion_nored


def optimizer_with_scheduler(optim_type, model, num_epochs, learning_rate, m=0.9, wd=5e-4):
    if optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=m, weight_decay=wd)
    elif optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_type == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing / 3600


from scipy.signal import lfilter


def filter(y):
    n = 1  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, y)
    return yy


"""
#General Training Loop with Data Selection Strategies
"""


def train_model(num_epochs, dataset_name, datadir, feature, model_name, fraction, select_every, optim_type, learning_rate, run,
                device, log_dir, trn_batch_size, strategy):

    # Loading the Dataset
    trainset, validset, testset, num_cls = load_dataset_custom(datadir, dataset_name, feature)
    N = len(trainset)
    val_batch_size = 1000
    tst_batch_size = 1000

    # Creating the Data Loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                              shuffle=False, pin_memory=True)

    valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                            shuffle=False, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                             shuffle=False, pin_memory=True)

    # Budget for subset selection
    bud = int(fraction * N)
    print("Budget, fraction and N:", bud, fraction, N)

    # Subset Selection and creating the subset data loader
    start_idxs = np.random.choice(N, size=bud, replace=False)
    idxs = start_idxs
    data_sub = Subset(trainset, idxs)
    subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=trn_batch_size,
                                                   shuffle=False, pin_memory=True)

    # Variables to store accuracies
    gammas = torch.ones(len(idxs)).to(device)
    substrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    subtrn_acc = np.zeros(num_epochs)

    # Results logging file
    print_every = 3
    
    all_logs_dir = log_dir + '/' + str(uuid.uuid4())
    while os.path.exists(all_logs_dir):
        all_logs_dir = log_dir +  '/' + str(uuid.uuid4())
    print(all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])
    path_logfile = os.path.join(all_logs_dir, 'log.txt')
    logfile = open(path_logfile, 'w')
    exp_name = dataset_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
               '_selEvery:' + str(select_every) + '_variant' + '_runs' + str(run)
    print(exp_name)

    # Model Creation
    model = create_model(model_name, num_cls, device)
    model1 = create_model(model_name, num_cls, device)
    # Loss Functions
    criterion, criterion_nored = loss_function()

    # Getting the optimizer and scheduler
    optimizer, scheduler = optimizer_with_scheduler(optim_type, model, num_epochs, learning_rate)

    if strategy == 'GradMatch':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerClassPerGradient',
                                          False, lam=0.5, eps=1e-100)
    elif strategy == 'GradMatchPB':
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerBatch',
                                          False, lam=0, eps=1e-100)

    elif strategy == 'GradMatch-Explore':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerClassPerGradient',
                                          False, lam=0.5, eps=1e-100)
        # Random-Online Selection strategy
        rand_setf_model = RandomStrategy(trainloader, online=True)

    elif strategy == 'GradMatchPB-Explore':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerBatch',
                                          False, lam=0, eps=1e-100)
        # Random-Online Selection strategy
        rand_setf_model = RandomStrategy(trainloader, online=True)

    elif strategy == 'Random':
        # Random Selection strategy
        setf_model = RandomStrategy(trainloader, online=False)

    elif strategy == 'Random-Online':
        # Random-Online Selection strategy
        setf_model = RandomStrategy(trainloader, online=True)

    print("=======================================", file=logfile)
    kappa_epochs = int(0.5 * num_epochs)
    full_epochs = floor(kappa_epochs/int(fraction*100))

    for i in range(num_epochs):
        subtrn_loss = 0
        subtrn_correct = 0
        subtrn_total = 0
        subset_selection_time = 0


        if (strategy in ['GLISTER', 'GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']) and (
                ((i + 1) % select_every) == 0):
            start_time = time.time()
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            if strategy in ['CRAIG', 'CRAIGPB']:
                subset_idxs, gammas = setf_model.select(int(bud), clone_dict, 'lazy')
            else:
                subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
            model.load_state_dict(cached_state_dict)
            idxs = subset_idxs
            if strategy in ['GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']:
                gammas = torch.from_numpy(np.array(gammas)).to(device).to(torch.float32)
            subset_selection_time += (time.time() - start_time)

        elif (strategy in ['GLISTER-Explore', 'GradMatch-Explore', 'GradMatchPB-Explore', 'CRAIG-Explore',
                           'CRAIGPB-Explore']):
            start_time = time.time()
            if i < full_epochs:
                subset_idxs, gammas = rand_setf_model.select(int(bud))
                idxs = subset_idxs
                gammas = gammas.to(device)
            elif ((i % select_every == 0) and (i >= kappa_epochs)):
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                if strategy in ['CRAIG-Explore', 'CRAIGPB-Explore']:
                    subset_idxs, gammas = setf_model.select(int(bud), clone_dict, 'lazy')
                else:
                    subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
                model.load_state_dict(cached_state_dict)
                idxs = subset_idxs
                if strategy in ['GradMatch-Explore', 'GradMatchPB-Explore', 'CRAIG-Explore', 'CRAIGPB-Explore']:
                    gammas = torch.from_numpy(np.array(gammas)).to(device).to(torch.float32)
            subset_selection_time += (time.time() - start_time)

        print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
        data_sub = Subset(trainset, idxs)
        subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=trn_batch_size, shuffle=False,
                                                       pin_memory=True)

        model.train()
        batch_wise_indices = list(subset_trnloader.batch_sampler)
        if strategy in ['CRAIG', 'CRAIGPB', 'GradMatch', 'GradMatchPB']:
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                inputs, targets = inputs.to(device), targets.to(device,
                                                                non_blocking=True)  # targets can have non_blocking=True.
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = criterion_nored(outputs, targets)
                loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (gammas[batch_wise_indices[batch_idx]].sum())
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif strategy in ['CRAIGPB-Explore', 'CRAIG-Explore', 'GradMatch-Explore', 'GradMatchPB-Explore']:
            start_time = time.time()
            if i < full_epochs:
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device,
                                                                    non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()

            elif i >= kappa_epochs:
                for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                    inputs, targets = inputs.to(device), targets.to(device,
                                                                    non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    losses = criterion_nored(outputs, targets)
                    loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (
                        gammas[batch_wise_indices[batch_idx]].sum())
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time


        elif strategy in ['Full']:
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device,
                                                                non_blocking=True)  # targets can have non_blocking=True.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time
        
        scheduler.step()
        timing[i] = train_time + subset_selection_time
        # print("Epoch timing is: " + str(timing[i]))

        val_loss = 0
        val_correct = 0
        val_total = 0
        tst_correct = 0
        tst_total = 0
        tst_loss = 0
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

            for batch_idx, (inputs, targets) in enumerate(testloader):
                # print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()

        val_acc[i] = val_correct / val_total
        tst_acc[i] = tst_correct / tst_total
        subtrn_acc[i] = subtrn_correct / subtrn_total
        substrn_losses[i] = subtrn_loss
        val_losses[i] = val_loss
        print('Epoch:', i + 1, 'Validation Accuracy: ', val_acc[i], 'Test Accuracy: ', tst_acc[i], 'Train Accuracy:', subtrn_acc[i], 'Time: ', timing[i])
    print(strategy + " Selection Run---------------------------------")
    print("Final SubsetTrn:", subtrn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc.max())
    print("Test Data Loss and Accuracy:", tst_loss, tst_acc.max())
    print('-----------------------------------')

    # Results logging into the file
    print(strategy, file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy, "
    tst = "Test Accuracy, "
    trn = "Train Accuracy, "
    time_str = "Time, "

    for i in range(num_epochs):
        time_str = time_str + "," + str(timing[i])
        val = val + "," + str(val_acc[i])
        trn = trn + "," + str(subtrn_acc[i])
        tst = tst + "," + str(tst_acc[i])

    print(timing, file=logfile)
    print(val, file=logfile)
    print(trn, file=logfile)
    print(tst, file=logfile)

    omp_timing = np.array(timing)
    omp_cum_timing = list(generate_cumulative_timing(omp_timing))
    omp_tst_acc = list(filter(tst_acc))
    print("Total time taken by " + strategy + " = " + str(omp_cum_timing[-1]))
    logfile.close()
    return {'loss': -tst_acc.max(), 'max_val_acc':val_acc.max() , 'train_acc': subtrn_acc.max(), 'status':STATUS_OK}

def main():
    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = '/content/drive/MyDrive/CordsResults/' + strategy + '/' + data_name + '/warmstart/' + str(
        fraction) + '/' + dt_string
    subprocess.run(["mkdir", "-p", log_dir])
    path_logfile = os.path.join(log_dir, data_name + '.txt')
    main_logfile = open(path_logfile, 'w')

    def optimize(config):
        try:
            print("*****************************************\n\n")
            print("Hyperparameters: ", config, file=main_logfile)
            tst_acc = train_model(num_epochs, data_name, datadir, feature, model_name, fraction, select_every, config['optimizer'], config['lr'], 1, device, log_dir, config['trn_batch_size'],
                'GradMatch')
            return tst_acc
        except Exception as err:
            print("Error occurred: ", str(err))
            rtn = {'status': STATUS_FAIL}
            return rtn


    """#Training Arguments"""
   
    max_evals = nb_evals = 5
    
    try:
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")
    start_time = time.time()
    best = fmin(
        optimize,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    print("Best parameters: ", best)
    print("Best parameters: ", best, file=main_logfile)
    pickle.dump(trials, open("results.pkl", "wb"))
    print("Total time taken by hyperparameter optimization: ", time.time() - start_time)
    print("Total time taken by hyperparameter optimization: ", time.time() - start_time, file=main_logfile)
    print("Model name: ", model_name, file=main_logfile)
    print("Dataset: ", data_name, file=main_logfile)
    print("Fraction: ", fraction, file=main_logfile)
    print("Number of epochs: ", num_epochs, file=main_logfile)
    print("Max Evals: ", max_evals, file=main_logfile)
    print("\n\n Trials : ", file=main_logfile)
    for trial in trials:
        print(trial, file=main_logfile)

if __name__ == "__main__":
    main()