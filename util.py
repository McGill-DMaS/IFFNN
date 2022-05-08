import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import time
import pickle
from tqdm import tqdm
from dataset_generation import *


def load_pickle(fn,compressed=False):
    if not compressed:
        f = open(fn,'rb')
    else:
        f = bz2.BZ2File(fn, 'rb')
    try:
        res = pickle.load(f)
    except:
        f.close()
        if not compressed:
            f = bz2.BZ2File(fn, 'rb')
            res = pickle.load(f)
            f.close()
        else:
            f = open(fn,'rb')
            res = pickle.load(f)
            f.close()
            write_pickle(res,fn,True)
        return res
    f.close()
    
    
    return res

def write_pickle(obj,fn,compressed=False):
    if not compressed:
        f = open(fn,'wb')
    else:    
        f = bz2.BZ2File(fn, 'wb')
    pickle.dump(obj,f)
    f.close()
    return

def explain_multi(x,out,num_classes,n_channel = 1):
    results = []
    for i in range(len(x)):
        current = []
        sam = x[i]
        current.append(sam)
        for j in range(num_classes):
            weight = out[i][j]
            if n_channel == 1:
                current.append(weight)
            else:
                weight = weight.reshape(n_channel,-1)
                current.append(weight)
        results.append(current)
    return results
def explain_binary(x,out,n_channel = 1):
    results = []
    for i in range(len(x)):
        current = []
        sam = x[i]
        current.append(sam)
        weight = out[i]
        if n_channel != 1:
            weight = weight.reshape(n_channel,-1)
        current.append(-weight)
        current.append(weight)
        results.append(current)
    return results
    
def train(model,all_classes,num_epochs,learning_rate,root_models,setting_name,name,seed,train_loader,valid_loader,test_loader,need_reshape=True,print_info=False, bicls = False,logi_training=False):
    assert(not (bicls and logi_training))
    t1 = time.time()
    device = torch.device("cuda:0")
    num_classes = len(all_classes)
    model = model.to(device)
    if bicls or logi_training:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss()
    if logi_training:
        print("logi training multi-class classification")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    n_total_steps = len(train_loader)
    print(name+" on {} classes".format(num_classes))
    best_valid_acc =-1
    best_valid_test_acc =-1
    best_epoch = -1
    save_path = root_models+setting_name+"_"+name+"_seed"+str(seed)+"_model.pkl"
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            if need_reshape:
                images = images.reshape(-1, images.shape[1]*images.shape[2]*images.shape[3])
            images = images.to(device)
            if bicls:
                labels = labels.numpy()
            else:
                labels = labels.to(device)
            outputs = model(images)
            if bicls:
                predicted = np.round(F.sigmoid(outputs).detach().cpu().numpy()).reshape(-1)
                n_samples += labels.size
                n_correct += (predicted == labels).sum().item() 
            
            else:
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item() 
        acc = 100.0 * n_correct / n_samples
        if print_info:
            print(f'Accuracy of the network on the 10000 test images: {acc} %') 
    
    for epoch in tqdm(range(num_epochs)):
        n_samples = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):  
            if need_reshape:
                images = images.reshape(-1, images.shape[1]*images.shape[2]*images.shape[3])
            images = images.to(device)
            if bicls:
                labels = labels.float()
            if logi_training:
                labels = labels.reshape(-1,1)
                y_onehot = torch.FloatTensor(labels.shape[0], num_classes)
                y_onehot.zero_()
                y_onehot.scatter_(1, labels, 1)
                labels = y_onehot
            labels = labels.to(device)
            n_samples += labels.size(0)
            outputs = model(images)
            if bicls:
                loss = criterion(outputs, labels.reshape(outputs.shape))
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in valid_loader:
                if need_reshape:
                    images = images.reshape(-1, images.shape[1]*images.shape[2]*images.shape[3])
                images = images.to(device)
                if bicls:
                    labels = labels.numpy()
                else:
                    labels = labels.to(device)
                outputs = model(images)
                if bicls:
                    predicted = np.round(F.sigmoid(outputs).detach().cpu().numpy()).reshape(-1)
                    n_samples += labels.size
                    n_correct += (predicted == labels).sum().item() 
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item() 
            valid_acc = 100.0 * n_correct / n_samples
            if print_info:
                print(f'Epoch {epoch} Accuracy of the network on the 10000 valid images: {valid_acc} %') 
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                if need_reshape:
                    images = images.reshape(-1, images.shape[1]*images.shape[2]*images.shape[3])
                images = images.to(device)
                if bicls:
                    labels = labels.numpy()
                else:
                    labels = labels.to(device)
                outputs = model(images)
                if bicls:
                    predicted = np.round(F.sigmoid(outputs).detach().cpu().numpy()).reshape(-1)
                    n_samples += labels.size
                    n_correct += (predicted == labels).sum().item() 
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item() 
            test_acc = 100.0 * n_correct / n_samples
            if print_info:
                print(f'Epoch {epoch} Accuracy of the network on the 10000 test images: {test_acc} %') 
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_valid_test_acc = test_acc
            best_epoch = epoch
            torch.save({
            'model_state_dict': model.state_dict(),
            }, save_path)
    print("takes:",time.time()-t1,'seconds')
    return best_valid_test_acc,best_valid_acc,best_epoch

def model_n_param(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def get_data_loaders_created_dataset(batch_size,n_train,n_valid,n_test,n_dim,n_min_co,n_max_co,n_min_pattern,n_max_pattern,def_class,class_priority,pos_rate,seed):

    np.random.seed(seed)

    train_set, valid_set, test_set, patterns = gen_INBEN(n_train, n_valid, n_test, n_dim, n_min_co, n_max_co, n_min_pattern, n_max_pattern, def_class, class_priority, pos_rate, seed)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                               batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, 
                                              batch_size=batch_size) 
    test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                              batch_size=batch_size, 
                                              shuffle=False) 
                                              
    return train_loader,valid_loader,test_loader,patterns
    
def get_data_loaders(batch_size,all_classes):
    random_seed=0

    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                           transform=transforms.ToTensor(),  
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=False, 
                                              transform=transforms.ToTensor()) 
    if len(all_classes) != 10:
        new_train_dataset = []
        for i in range(len(train_dataset)):
            if train_dataset[i][1] in all_classes:
                new_train_dataset.append(train_dataset[i])
        new_test_dataset = []
        for i in range(len(test_dataset)):
            if test_dataset[i][1] in all_classes:
                new_test_dataset.append(test_dataset[i])
        train_dataset =  new_train_dataset    
        test_dataset = new_test_dataset        
    trainset_size = len(train_dataset)
    indices = list(range(trainset_size))
    
    if indices == 60000:    
        split = 50000
    else:
        split = int(5/6*trainset_size)
        
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    print("n training samples:",len(train_indices))
    print("n valid samples:",len(val_indices))
    print("n test samples:",len(test_dataset))
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               #shuffle=True,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                              batch_size=batch_size, 
                                              #shuffle=False,
                                              sampler=valid_sampler) 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False) 
    return train_loader,valid_loader,test_loader