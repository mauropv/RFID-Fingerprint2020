import numpy as np
from utils import *
import torch.nn as nn
import torch.nn.functional as F


np.set_printoptions(suppress=True)
np.random.seed(1)


def load_data_split(root_path, start, end, first_id, last_id, letts, wave_piece_size):

    X, y = [], []

    for raw_signal_file in os.listdir(root_path):
        tag_id = raw_signal_file[5:8]
        tag_let = raw_signal_file[8]
        if (tag_let not in letts or first_id > int(tag_id) or last_id <= int(tag_id)):
            continue
        print(tag_id, tag_let)
        signal_loaded = np.load(open(root_path + raw_signal_file, "rb"))
        for wave_id in range(int(len(signal_loaded) * start), int(len(signal_loaded) * end)):
            wave = signal_loaded[wave_id]
            for i in range(0, len(wave) - wave_piece_size, 1024):


                x_tmp = np.asarray(wave[i:i + wave_piece_size])
                X += [x_tmp]
                y += [int(tag_id)]

    X = np.asarray(X)
    y = np.asarray(y)
    labels = np.unique(y)

    return X, y, labels


def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())


def client_update(client_model, optimizer, train_loader, epoch):
    criterion = nn.CrossEntropyLoss()
    """
    This function updates/trains client model on client data
    """
    client_model.train()
    accs = []
    for e in range(epoch):
        running_accuracy = 0.

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output = client_model(data)
            loss = criterion(output, target.long())

            # loss = F.nll_loss(output, target.long())
            loss.backward()
            optimizer.step()
            running_accuracy += accuracy(output, target) / len(train_loader)
        print(str(running_accuracy.cpu().numpy()))
        accs.append(running_accuracy.cpu().numpy())
        # print(accs)

    return loss.item(), accs


def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    for m in client_models:
        m.load_state_dict(global_model.state_dict())


def test(testing_model, testing_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    testing_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testing_loader:
            data, target = data.cuda(), target.cuda()
            output = testing_model(data)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testing_loader.dataset)
    acc = correct / len(testing_loader.dataset)
    del testing_model

    return test_loss, acc


def augment(data, target):
    pitch = [5,10,20,100]
    data_ret, target_ret = data.clone(), target.clone()
    means = data.mean(axis = 2)
    for p in pitch:
        dr,tr = data.clone(), target.clone()
        i = -1
        for tensor in dr:
            i += 1
            tensor[:,0] = tensor[:,0].add(torch.normal(0, abs(means[i][0]/p), size=(1, data.shape[1])))
            tensor[:,1] = tensor[:,1].add(torch.normal(0, abs(means[i][1]/p), size=(1, data.shape[1])))
        data_ret = torch.cat((data_ret, dr), dim = 0)
        target_ret = torch.cat((target_ret, tr), dim = 0)
    return data_ret, target_ret