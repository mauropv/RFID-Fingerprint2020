import os
import socket
import numpy as np
import torch
from utils import *
import torch.nn as nn
np.set_printoptions(suppress=True)
np.random.seed(1)
import torch.optim as optim
import copy
import pickle

PATH_TRAIN = "/tmp/"
PATH_VALID = "/tmp/"
PATH_MODELS = "/tmp/"
PATH_RESULTS = "/tmp/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 64

def generateTrainSet(lettera, pdu, num_tags, PATH_TRAIN, params, augment):
    X, y, labels = load_data_split(PATH_TRAIN, 0, pdu, 0, num_tags, lettera)
    X_train = torch.from_numpy(X).float()
    X_train = X_train.transpose(1, 2).contiguous()
    y_train = torch.from_numpy(y)
    if augment :
      X_train, y_train = augment(X_train, y_train)
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    training_generator = torch.utils.data.DataLoader(dataset, **params)
    return training_generator


def do_exp(trainA, trainB, trainC, test_set, agents, pdu, pretrained, num_tags, do_augment):
    kernel_size = 3

    path_results = PATH_RESULTS+"ta_"+trainA+"_tb_"+trainB+"_tc_"+trainC+"_t_"+test_set+"_a_"+str(agents)+"_nt_"+str(num_tags)+"_pdu_"+str(pdu)+"_pr_"+str(pretrained)+"/"
    path_results_models = PATH_RESULTS+"ta_" + trainA + "_tb_" + trainB + "_tc_" + trainC + "_t_" + test_set + "_a_" + str(agents) + "_nt_" + str(num_tags) + "_pdu_" + str(pdu) + "_pr_" + str(pretrained) + "/models/"

    try:
        os.mkdir(path_results)
        os.mkdir(path_results_models)
    except OSError:
        print("Creation of the directory %s failed" % path_results)


    a0 = nn.Conv1d(in_channels=2, out_channels=25, kernel_size=kernel_size, padding=(kernel_size // 2))
    a1 = nn.LeakyReLU(negative_slope=0.1)
    b0 = nn.MaxPool1d(kernel_size=2, padding=1)

    c0 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=kernel_size, padding=(kernel_size // 2))
    c1 = nn.LeakyReLU(negative_slope=0.1)
    d0 = nn.MaxPool1d(2, padding=1)

    ca0 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=kernel_size, padding=(kernel_size // 2))
    ca1 = nn.LeakyReLU(negative_slope=0.1)
    ca2 = nn.MaxPool1d(2, padding=1)

    e0 = nn.Flatten()
    e1 = nn.Linear(6425, num_tags) #6425


    # Example of a model with 3 convs
    #model = nn.Sequential(a0,a1,b0,c0,c1,d0,ca0,ca1,ca2,e0,e1)
    model = nn.Sequential(a0,a1,b0,c0,c1,d0,e0,e1)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    params = {'batch_size': batch_size,
          'shuffle': True}


    train_loader = [generateTrainSet(trainA, pdu, num_tags, PATH_TRAIN, params, do_augment), generateTrainSet(trainB, pdu, num_tags, PATH_TRAIN, params,do_augment), generateTrainSet(trainC, pdu, num_tags, PATH_TRAIN, params, do_augment)]

    Xt, yt, labelst = load_data_split(PATH_VALID, 0, 1, 0, num_tags, test_set)

    X_test = torch.from_numpy(Xt).float()
    X_test = X_test.transpose(1, 2).contiguous()
    y_test = torch.from_numpy(yt)
    dataset_t = torch.utils.data.TensorDataset(X_test, y_test)

    test_generator = torch.utils.data.DataLoader(dataset_t, **params)
    test_loader = test_generator

    num_clients = agents
    num_selected = agents
    num_rounds = 100
    epochs = 1

    ############################################
    #### Initializing models and optimizer  ####
    ############################################
    if pretrained:
        model = torch.load("path/pretrained/model")
    # model = net
    #### global model ##########
    global_model = copy.deepcopy(model).cuda()

    ############## client models ##############
    client_models = [copy.deepcopy(model).cuda() for _ in range(num_selected)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict())  ### initial synchronizing with global model

    ############### optimizers ################
    opt = [optim.Adam(model.parameters()) for model in client_models]


    ###### List containing info about learning #########
    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []
    # Runnining FL

    accs_tot = [[], [], []]

    for r in range(num_rounds):
        # select random clients
        client_idx = np.random.permutation(num_clients)[:num_selected]
        # client update
        loss = 0
        for i in range(num_selected):
            print("CLI", client_idx[i])
            l, accs = client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs)
            loss += l
            for a in accs:
                accs_tot[client_idx[i]].append(a)

        losses_train.append(loss)
        # server aggregate
        for i in range(num_selected):
            t, a = test(client_models[i], test_loader)
            # print(mod.state_dict())
            print("Wewe", t, a)

        server_aggregate(global_model, client_models)
        test_loss, acc = test(global_model, test_loader)
        torch.save(global_model, path_results_models+"model"+str(r))
        losses_test.append(test_loss)
        acc_test.append(acc)
        print('%d-th round' % r)
        print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))

    tf = open(path_results+'acc_test.txt', "w")
    for i in acc_test:
        print(i)
        tf.write(str(i)+"\n")
    tf.close()

    tf = open(path_results+'acc_train.txt', "w")
    for line in range(len(accs_tot[0])):
        str_int = ""
        for col in range(len(accs_tot)):
            str_int += str(accs_tot[0][line]) +"\t"
        tf.write(str_int+"\n")
        print(str_int)
    tf.close()

    with open(path_results+'acc_test', 'wb') as fp:
        pickle.dump(acc_test, fp)

    with open(path_results+'acc_train', 'wb') as fp:
        pickle.dump(accs_tot, fp)



#pdu = 0.1
#tags = 200
#Experiments examples
#A su A
#do_exp("A","A","A","A",1,pdu,False,tags)
#B su B
#do_exp("B","A","A","B",1,pdu,False,tags)
#A su B
#do_exp("A","A","A","B",1,pdu,False,tags)
#B su A
#do_exp("B","A","A","A",1,pdu,False,tags)
#ABC su ABC
#do_exp("ABC","A","A","ABC",1,pdu,False,tags)

#do_exp("C","A","A","C",1,pdu,False,tags)
#Fed classic

#Pameters:
#1) Client 1 dataset
#2) Client 2 dataset
#3) Client 3 dataset
#4) Test Dataset
#5) Number of Active Clients
#6) Percentual of data to load for train
#7) If True uses a pretrained model
#8) Number of tags to use in the test
#9) Activate Data Augmentation
#do_exp( "A", "A", "A", "A", 1, pdu, False, tags, False)

#Fed pretrained
