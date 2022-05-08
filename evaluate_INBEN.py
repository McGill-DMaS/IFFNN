from model import *
import time
import os


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

models = [SoftmaxRegression, FFNN, IFFNN, FFNN, FFNN, HWNet, HW_IFFNN, HWNet, HWNet]
model_names = ["SR","FC-MC1","FC-IFFNN-MC","FC-MC2","FC-MC3","HW-MC1","HW-IFFNN-MC","HW-MC2","HW-MC3"]
hyp_params = [(input_size, num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [int(3*hidden_size)]*3, num_classes),(input_size, [int(2.21*hidden_size)]*5, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [int(1.815*hidden_size)]*3, num_classes),(input_size, [int(1.27*hidden_size)]*6, num_classes)]

seeds = [0,1,2,3,4]
setting_name = 'createddataset10cls'
assert(len(models)==len(model_names))
assert(len(hyp_params)==len(model_names))

train_loader,valid_loader,test_loader,patterns = get_data_loaders_created_dataset(batch_size,n_train,n_valid,n_test,n_dim,n_min_co,n_max_co,n_min_pattern,n_max_pattern,def_class,class_priority,pos_rate,seed)

results = {}
root_res = './results/'
root_models = './models/'
if not os.path.exists(root_res):
    os.mkdir(root_res)
if not os.path.exists(root_models):
    os.mkdir(root_models)

f = open("result_inben.txt",'a')
f.write(setting_name+"\n")
f.close()
for mod, name, hyp in zip(models,model_names,hyp_params):
    print("name:",name)
    save_path = root_res+setting_name+"_"+name+"_res.pkl"
    if os.path.exists(save_path):
        print(mod,"model done")
        results[name] = load_pickle(save_path)
        print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
        f = open("result_inben.txt",'a')
        f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
        f.close()
        continue
    logi_training = False
    for seed in seeds:
        need_reshape = False
        torch.manual_seed(seed)
        net = mod(*hyp)
        n_param = model_n_param(net)
        t1 = time.time()
        best_valid_test_acc,best_valid_acc,best_epoch = train(net,all_classes,num_epochs,learning_rate,root_models,setting_name,name,seed,train_loader,valid_loader,test_loader,need_reshape,print_info=False,logi_training=logi_training)
        t2 = time.time()
        consumption = t2-t1
        print(name,"takes",consumption,"seconds")        
        if name not in results:
            results[name] = {'n_param':n_param,'valid_acc':[],'best_epoch':[],'test_acc':[],'time':[]}
        results[name]['test_acc'].append(best_valid_test_acc)
        results[name]['valid_acc'].append(best_valid_acc)
        results[name]['time'].append(consumption)
        results[name]['best_epoch'].append(best_epoch)
    print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
    f = open("result_inben.txt",'a')
    f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
    f.close()
    write_pickle(results[name],save_path)


for method, result in results.items():
    print(method, "n_param:",result['n_param']/1000/1000, "valid acc avg:", np.mean(result['valid_acc']), "test acc avg:", np.mean(result['test_acc']),'best epoch:', np.mean(result['best_epoch']))
    

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
num_classes = len(class_priority)
input_size = n_dim

all_classes = set(class_priority)

models = [SoftmaxRegression, FFNN, IFFNN, FFNN, FFNN, HWNet, HW_IFFNN, HWNet, HWNet]
model_names = ["SR","FC-MC1","FC-IFFNN-MC","FC-MC2","FC-MC3","HW-MC1","HW-IFFNN-MC","HW-MC2","HW-MC3"]
hyp_params = [(input_size, num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [int(1.56*hidden_size)]*3, num_classes),(input_size, [int(1.19*hidden_size)]*5, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [int(1.205*hidden_size)]*3, num_classes),(input_size, [int(0.93*hidden_size)]*5, num_classes)]

seeds = [0,1,2,3,4]
setting_name = 'createddataset2cls'
assert(len(models)==len(model_names))
assert(len(hyp_params)==len(model_names))
train_loader,valid_loader,test_loader,patterns = get_data_loaders_created_dataset(batch_size,n_train,n_valid,n_test,n_dim,n_min_co,n_max_co,n_min_pattern,n_max_pattern,def_class,class_priority,pos_rate,seed)



results = {}
root_res = './results/'
root_models = './models/'
f = open("result_inben.txt",'a')
f.write(setting_name+"\n")
f.close()
for mod, name, hyp in zip(models,model_names,hyp_params):
    print("name:",name)
    save_path = root_res+setting_name+"_"+name+"_res.pkl"
    if os.path.exists(save_path):
        print(mod,"model done")
        results[name] = load_pickle(save_path)
        print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
        f = open("result_inben.txt",'a')
        f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
        f.close()
        continue
    for seed in seeds:
        need_reshape = False
        torch.manual_seed(seed)
        net = mod(*hyp)
        n_param = model_n_param(net)
        t1 = time.time()
        best_valid_test_acc,best_valid_acc,best_epoch = train(net,all_classes,num_epochs,learning_rate,root_models,setting_name,name,seed,train_loader,valid_loader,test_loader,need_reshape)
        t2 = time.time()
        consumption = t2-t1
        print(name,"takes",consumption,"seconds")        
        if name not in results:
            results[name] = {'n_param':n_param,'valid_acc':[],'best_epoch':[],'test_acc':[],'time':[]}
        results[name]['test_acc'].append(best_valid_test_acc)
        results[name]['valid_acc'].append(best_valid_acc)
        results[name]['time'].append(consumption)
        results[name]['best_epoch'].append(best_epoch)
    print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
    f = open("result_inben.txt",'a')
    f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
    f.close()
    write_pickle(results[name],save_path)


for method, result in results.items():
    print(method, "n_param:",result['n_param']/1000/1000, "valid acc avg:", np.mean(result['valid_acc']), "test acc avg:", np.mean(result['test_acc']),'best epoch:', np.mean(result['best_epoch']))

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
num_classes = len(class_priority)
input_size = n_dim

models = [LogisticRegression, FFNN, IFFNN, FFNN, FFNN, HWNet, HW_IFFNN, HWNet, HWNet]
model_names = ["LR","FC-BC1","FC-IFFNN-BC","FC-BC2","FC-BC3","HW-BC1","HW-IFFNN-BC","HW-BC2","HW-BC3"]
hyp_params = [(input_size,),(input_size, [hidden_size,hidden_size,hidden_size], num_classes, True),(input_size, [hidden_size,hidden_size,hidden_size], num_classes, True),(input_size, [int(1.3*hidden_size)]*3, num_classes, True),(input_size, [int(1.0*hidden_size)]*5, num_classes, True),(input_size, [hidden_size]*3, num_classes, True),(input_size, [hidden_size]*3, num_classes, True),(input_size, [int(1.105*hidden_size)]*3, num_classes, True),(input_size, [int(0.857*hidden_size)]*5, num_classes, True)]


train_loader,valid_loader,test_loader,patterns = get_data_loaders_created_dataset(batch_size,n_train,n_valid,n_test,n_dim,n_min_co,n_max_co,n_min_pattern,n_max_pattern,def_class,class_priority,pos_rate,seed)



results = {}
root_res = './results/'
root_models = './models/'
f = open("result_inben.txt",'a')
f.write(setting_name+"\n")
f.close()
for mod, name, hyp in zip(models,model_names,hyp_params):
    print("name:",name)
    save_path = root_res+setting_name+"_"+name+"_res.pkl"
    if os.path.exists(save_path):
        print(mod,"model done")
        results[name] = load_pickle(save_path)
        print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
        f = open("result_inben.txt",'a')
        f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
        f.close()
        continue
    for seed in seeds:
        need_reshape = False
        torch.manual_seed(seed)
        net = mod(*hyp)
        n_param = model_n_param(net)
        t1 = time.time()
        best_valid_test_acc,best_valid_acc,best_epoch = train(net,all_classes,num_epochs,learning_rate,root_models,setting_name,name,seed,train_loader,valid_loader,test_loader,need_reshape, bicls = True)
        t2 = time.time()
        consumption = t2-t1
        print(name,"takes",consumption,"seconds")        
        if name not in results:
            results[name] = {'n_param':n_param,'valid_acc':[],'best_epoch':[],'test_acc':[],'time':[]}
        results[name]['test_acc'].append(best_valid_test_acc)
        results[name]['valid_acc'].append(best_valid_acc)
        results[name]['time'].append(consumption)
        results[name]['best_epoch'].append(best_epoch)
    print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
    f = open("result_inben.txt",'a')
    f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
    f.close()
    write_pickle(results[name],save_path)



for method, result in results.items():
    print(method, "n_param:",result['n_param']/1000/1000, "valid acc avg:", np.mean(result['valid_acc']), "test acc avg:", np.mean(result['test_acc']),'best epoch:', np.mean(result['best_epoch']))
