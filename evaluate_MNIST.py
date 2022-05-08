from model import *
import time
import os


input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
batch_size = 256
num_epochs = 200
learning_rate = 0.001 



all_classes =  [0,1,2,3,4,5,6,7,8,9]
num_classes = len(all_classes)
models = [SoftmaxRegression, FFNN, IFFNN, FFNN, FFNN, CNNNet, I_CNNNet, CNNNet2, CNNNet3, ResNet, ResNet_IFFNN, ResNet2, ResNet3, HWNet, HW_IFFNN, HWNet, HWNet]
model_names = ["SR","FC-MC1","FC-IFFNN-MC","FC-MC2","FC-MC3","CNN-MC1","CNN-IFFNN-MC","CNN-MC2","CNN-MC3","ResNET-MC1","ResNET-IFFNN-MC","ResNET-MC2","ResNET-MC3","HW-MC1","HW-IFFNN-MC","HW-MC2","HW-MC3"]
hyp_params = [(input_size, num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [int(2.73*hidden_size)]*3, num_classes),(input_size, [2*hidden_size]*5, num_classes),(),(),(),(),(ResidualBlock,[2, 2, 2, 2]),(ResidualBlock,[2, 2, 2, 2]),(ResidualBlock,[2, 2, 2, 2]),(ResidualBlock,[2, 2, 2, 2]),(input_size, [hidden_size]*3, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [int(1.682*hidden_size)]*3, num_classes),(input_size, [int(1.17*hidden_size)]*6, num_classes)]

seeds = [0,1,2,3,4]
setting_name = '10cls'
assert(len(models)==len(model_names))
assert(len(hyp_params)==len(model_names))



train_loader,valid_loader,test_loader = get_data_loaders(batch_size,all_classes)
results = {}
root_res = './results/'
root_models = './models/'

if not os.path.exists(root_res):
    os.mkdir(root_res)
if not os.path.exists(root_models):
    os.mkdir(root_models)

f = open("result_mnist.txt",'a')
f.write(setting_name+"\n")
f.close()
for mod, name, hyp in zip(models,model_names,hyp_params):
    print("name:",name)
    save_path = root_res+setting_name+"_"+name+"_res.pkl"
    if os.path.exists(save_path):
        print(mod,"model done")
        results[name] = load_pickle(save_path)
        print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
        f = open("result_mnist.txt",'a')
        f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
        f.close()
        continue
    for seed in seeds:
        if "CNN" in name or "ResNET" in name:
            print("model:",name,"doesn't need reshape")
            need_reshape = False
        else:
            print("model:",name,"needs reshape")
            need_reshape = True
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
    f = open("result_mnist.txt",'a')
    f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
    f.close()
    write_pickle(results[name],save_path)




for method, result in results.items():
    print(method, "valid acc avg:", np.mean(result['valid_acc']), "test acc avg:", np.mean(result['test_acc']),'best epoch:', np.mean(result['best_epoch']))
    
all_classes =  [0,1]
num_classes = len(all_classes)
models = [SoftmaxRegression, FFNN, IFFNN, FFNN, FFNN, CNNNet, I_CNNNet, CNNNet2_MC2, CNNNet2_MC3, ResNet, ResNet_IFFNN, ResNet2_MC2, ResNet2_MC3, HWNet, HW_IFFNN, HWNet, HWNet]
model_names = ["SR","FC-MC1","FC-IFFNN-MC","FC-MC2","FC-MC3","CNN-MC1","CNN-IFFNN-MC","CNN-MC2","CNN-MC3","ResNET-MC1","ResNET-IFFNN-MC","ResNET-MC2","ResNET-MC3","HW-MC1","HW-IFFNN-MC","HW-MC2","HW-MC3"]
hyp_params = [(input_size, num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [int(1.481*hidden_size)]*3, num_classes),(input_size, [int(1.11*hidden_size)]*5, num_classes),(num_classes,),(num_classes,),(num_classes,),(num_classes,),(ResidualBlock,[2, 2, 2, 2],num_classes),(ResidualBlock,[2, 2, 2, 2],num_classes),(ResidualBlock,[2, 2, 2, 2],num_classes),(ResidualBlock,[2, 2, 2, 2],num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [int(1.167*hidden_size)]*3, num_classes),(input_size, [449]*5, num_classes)]

seeds = [0,1,2,3,4]
setting_name = '2cls01'
assert(len(models)==len(model_names))
assert(len(hyp_params)==len(model_names))



train_loader,valid_loader,test_loader = get_data_loaders(batch_size,all_classes)
results = {}
root_res = './results/'
root_models = './models/'
f = open("result_mnist.txt",'a')
f.write(setting_name+"\n")
f.close()
for mod, name, hyp in zip(models,model_names,hyp_params):
    save_path = root_res+setting_name+"_"+name+"_res.pkl"
    if os.path.exists(save_path):
        print(mod,"model done")
        results[name] = load_pickle(save_path)
        print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
        f = open("result_mnist.txt",'a')
        f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
        f.close()
        continue
    for seed in seeds:
        if "CNN" in name or "ResNET" in name:
            print("model:",name,"doesn't need reshape")
            need_reshape = False
        else:
            print("model:",name,"needs reshape")
            need_reshape = True
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
    f = open("result_mnist.txt",'a')
    f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
    f.close()
    write_pickle(results[name],save_path)


for method, result in results.items():
    print(method, "valid acc avg:", np.mean(result['valid_acc']), "test acc avg:", np.mean(result['test_acc']),'best epoch:', np.mean(result['best_epoch']))
    
    
    
    
    
    
    
    
    
all_classes =  [0,1]
num_classes = len(all_classes)
models = [LogisticRegression,FFNN,IFFNN,FFNN,FFNN, CNNNet, I_CNNNet, CNNNet_BC2, CNNNet_BC3,ResNet,ResNet_IFFNN,ResNet_BC2,ResNet_BC3,HWNet,HW_IFFNN,HWNet,HWNet]
model_names = ["LR","FC-BC1","FC-IFFNN-BC","FC-BC2","FC-BC3","CNN-BC1","CNN-IFFNN-BC","CNN-BC2","CNN-BC3","ResNET-BC1","ResNET-IFFNN-BC","ResNET-BC2","ResNET-BC3","HW-BC1","HW-IFFNN-BC","HW-BC2","HW-BC3"]
hyp_params = [(input_size,),(input_size, [hidden_size,hidden_size,hidden_size], num_classes, True),(input_size, [hidden_size,hidden_size,hidden_size], num_classes, True),(input_size, [int(1.26*hidden_size)]*3, num_classes, True),(input_size, [int(1.08*hidden_size)]*4, num_classes, True),(num_classes, True),(num_classes, True),(num_classes, True),(num_classes, True),(ResidualBlock,[2, 2, 2, 2], num_classes, True),(ResidualBlock,[2, 2, 2, 2], num_classes, True),(ResidualBlock,[2, 2, 2, 2], num_classes, True),(ResidualBlock,[2, 2, 2, 2], num_classes, True),(input_size, [hidden_size]*3, num_classes, True),(input_size, [hidden_size]*3, num_classes, True),(input_size, [int(1.085*hidden_size)]*3, num_classes, True),(input_size, [int(0.936*hidden_size)]*4, num_classes, True)]

seeds = [0,1,2,3,4]
setting_name = '2cls01'
assert(len(models)==len(model_names))
assert(len(hyp_params)==len(model_names))



train_loader,valid_loader,test_loader = get_data_loaders(batch_size,all_classes)
results = {}
root_res = './results/'
root_models = './models/'
f = open("result_mnist.txt",'a')
f.write(setting_name+"\n")
f.close()
for mod, name, hyp in zip(models,model_names,hyp_params):
    save_path = root_res+setting_name+"_"+name+"_res.pkl"
    if os.path.exists(save_path):
        print(mod,"model done")
        results[name] = load_pickle(save_path)
        print(name, "n_param:",results[name]['n_param']/1000/1000, "valid acc avg:", np.mean(results[name]['valid_acc']), "test acc avg:", np.mean(results[name]['test_acc']),'best epoch:', np.mean(results[name]['best_epoch']))
        f = open("result_mnist.txt",'a')
        f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
        f.close()
        continue
    for seed in seeds:
        if "CNN" in name or "ResNET" in name:
            print("model:",name,"doesn't need reshape")
            need_reshape = False
        else:
            print("model:",name,"needs reshape")
            need_reshape = True
        torch.manual_seed(seed)
        net = mod(*hyp)
        n_param = model_n_param(net)
        t1 = time.time()
        best_valid_test_acc,best_valid_acc,best_epoch = train(net,all_classes,num_epochs,learning_rate,root_models,setting_name,name,seed,train_loader,valid_loader,test_loader,need_reshape,bicls=True)
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
    f = open("result_mnist.txt",'a')
    f.write("{} {}\n".format(name,np.mean(results[name]['test_acc'])))
    f.close()
    write_pickle(results[name],save_path)


for method, result in results.items():
    print(method, "valid acc avg:", np.mean(result['valid_acc']), "test acc avg:", np.mean(result['test_acc']),'best epoch:', np.mean(result['best_epoch']))