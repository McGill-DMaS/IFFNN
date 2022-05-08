from model import *
from sklearn.model_selection import ParameterGrid
from sklearn import tree


def evaluate(train_x,train_y,valid_x,valid_y,test_x,test_y):
    print("n train: {} n valid: {} n test: {}".format(len(train_x),len(valid_x),len(test_x)))
    print("dimension:",len(train_x[0]))
    results = []
    for random_seed in [0,1,2,3,4]:
        print("seed:",random_seed)
        np.random.seed(random_seed)
        best_score = 0.
        for g in ParameterGrid(param_grid_DT):
            clf = clf_()
            clf.set_params(**g)
            clf.fit(train_x,train_y)
            tmp_score = clf.score(valid_x,valid_y)
            if tmp_score > best_score:
                best_score = tmp_score
                best_grid = g
                best_clf = clf
                print("new nest valid score:",tmp_score,'with:',g)
        clf = best_clf
        tmp_score = clf.score(test_x,test_y)
        print("test score:",tmp_score)
        results.append(tmp_score)
    print("avg test acc:",np.mean(results))



print("=========\n 10-class MNIST:")

input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 200
learning_rate = 0.001 
#plt.axis('off')

dataset = "MNIST"
random_seed=0
all_classes =  [0,1,2,3,4,5,6,7,8,9]
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                       transform=transforms.ToTensor(),  
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor()) 
                                          
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
train_set = [train_dataset[ind] for ind in train_indices]
valid_set = [train_dataset[ind] for ind in val_indices]
test_set = test_dataset


clf_ = tree.DecisionTreeClassifier

param_grid_DT = [
          {'criterion': ['gini','entropy'], 'max_depth': [10,25,50,100,200,300,400,500,1000]}]
          
train_x = [sam[0].reshape(-1).tolist() for sam in train_set]
train_y = [sam[1] for sam in train_set]
valid_x = [sam[0].reshape(-1).tolist() for sam in valid_set]
valid_y = [sam[1] for sam in valid_set]
test_x = [sam[0].reshape(-1).tolist() for sam in test_set]
test_y = [sam[1] for sam in test_set]

evaluate(train_x,train_y,valid_x,valid_y,test_x,test_y)










print("=========\n 2-class MNIST:")

dataset = "MNIST"
random_seed=0
all_classes =  [0,1]
if dataset == "MNIST":
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                           transform=transforms.ToTensor(),  
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=False, 
                                              transform=transforms.ToTensor()) 
else:
    assert(dataset == "CIFAR")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True, 
                                           transform=transforms.ToTensor(),  
                                               download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                              train=False, 
                                              transform=transforms.ToTensor()) 
    print("CIFAR train {} samples, test {} samples".format(len(train_dataset),len(test_dataset)))

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

if dataset == "MNIST":
    if indices == 60000:    
        split = 50000
    else:
        split = int(5/6*trainset_size)
else:
    assert(dataset == "CIFAR")
    if indices == 50000:    
        split = 40000
    else:
        split = int(4/5*trainset_size)
    
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]
train_set = [train_dataset[ind] for ind in train_indices]
valid_set = [train_dataset[ind] for ind in val_indices]
test_set = test_dataset

train_x = [sam[0].reshape(-1).tolist() for sam in train_set]
train_y = [sam[1] for sam in train_set]
valid_x = [sam[0].reshape(-1).tolist() for sam in valid_set]
valid_y = [sam[1] for sam in valid_set]
test_x = [sam[0].reshape(-1).tolist() for sam in test_set]
test_y = [sam[1] for sam in test_set]

evaluate(train_x,train_y,valid_x,valid_y,test_x,test_y)



print("=========\n 10-class INBEN:")

batch_size = 256
hidden_size = 500 
n_train=100000
n_valid=10000
n_test=10000
n_dim=1000
n_min_co=2
n_max_co=5
n_min_pattern=7
n_max_pattern=13
def_class=0
pos_rate=0.03
num_epochs = 200
learning_rate = 0.001 
seed=0
class_priority=[6,4,9,3,5,7,1,2,0,8]
num_classes = len(class_priority)
input_size = n_dim
all_classes = set(class_priority)

np.random.seed(seed)

train_set, valid_set, test_set, patterns = gen_INBEN(n_train, n_valid, n_test, n_dim, n_min_co, n_max_co, n_min_pattern, n_max_pattern, def_class, class_priority, pos_rate, seed)

train_x = [sam[0].tolist() for sam in train_set]
train_y = [sam[1] for sam in train_set]
valid_x = [sam[0].tolist() for sam in valid_set]
valid_y = [sam[1] for sam in valid_set]
test_x = [sam[0].tolist() for sam in test_set]
test_y = [sam[1] for sam in test_set]

evaluate(train_x,train_y,valid_x,valid_y,test_x,test_y)

print("=========\n 2-class INBEN:")

batch_size = 256
hidden_size = 500 
n_train=20000
n_valid=2000
n_test=2000
n_dim=1000
n_min_co=2
n_max_co=5
n_min_pattern=7
n_max_pattern=13
def_class=0
pos_rate=0.03
num_epochs = 200
learning_rate = 0.001 
seed=0
class_priority=[1,0]
#class_priority=[6,4,9,3,5,7,1,2,0,8]
num_classes = len(class_priority)
input_size = n_dim

all_classes = set(class_priority)

np.random.seed(seed)

train_set, valid_set, test_set, patterns = gen_INBEN(n_train, n_valid, n_test, n_dim, n_min_co, n_max_co, n_min_pattern, n_max_pattern, def_class, class_priority, pos_rate, seed)

train_x = [sam[0].tolist() for sam in train_set]
train_y = [sam[1] for sam in train_set]
valid_x = [sam[0].tolist() for sam in valid_set]
valid_y = [sam[1] for sam in valid_set]
test_x = [sam[0].tolist() for sam in test_set]
test_y = [sam[1] for sam in test_set]



evaluate(train_x,train_y,valid_x,valid_y,test_x,test_y)