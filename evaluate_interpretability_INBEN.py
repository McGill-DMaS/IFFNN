from model import *
import operator
import progressbar


seed=0
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
class_priority=[6,4,9,3,5,7,1,2,0,8]
num_classes = len(class_priority)
input_size = n_dim
all_classes = set(class_priority)

models = [SoftmaxRegression,FFNN,IFFNN,FFNN, FFNN,HWNet,HW_IFFNN,HWNet,HWNet]
model_names = ["SR","FC-MC1","FC-IFFNN-MC","FC-MC2","FC-MC3","HW-MC1","HW-IFFNN-MC","HW-MC2","HW-MC3"]
hyp_params = [(input_size, num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [int(2.73*hidden_size)]*3, num_classes),(input_size, [2*hidden_size]*5, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [int(1.682*hidden_size)]*3, num_classes),(input_size, [int(1.17*hidden_size)]*6, num_classes)]

seeds = [0,1,2,3,4]
setting_name = 'createddataset10cls'
assert(len(models)==len(model_names))
assert(len(hyp_params)==len(model_names))

train_loader,valid_loader,test_loader,patterns = get_data_loaders_created_dataset(batch_size,n_train,n_valid,n_test,n_dim,n_min_co,n_max_co,n_min_pattern,n_max_pattern,def_class,class_priority,pos_rate,seed)
device = torch.device("cuda:0")


print("Interpretability evaluation on 10 classes:")

bicls = False

root_res = './results/'
root_models = './models/'
seeds = [0,1,2,3,4]
for mod, name, hyp in zip(models,model_names,hyp_params):
    if "IFFNN" not in name and 'SR' not in name:
        continue
    print("name:",name)
    in_results = []
    for seed in seeds:
        need_reshape = False
        
        save_path = root_models+setting_name+"_"+name+"_seed"+str(seed)+"_model.pkl"
        net = mod(*hyp)
        checkpoint = torch.load(save_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
        net.eval()
        results = []
        picked_labels = []
        picked_picked_outputs = []
        all_samples = 0
        n_correct = 0
        for samples, labels in test_loader:
            samples = samples.to(device)
            if bicls:
                labels = labels.numpy()
            else:
                labels = labels.to(device)
            outputs = net(samples)
            explains = net.explain(samples)
            
            if bicls:
                predicted = np.round(outputs.detach().cpu().numpy())
            
            else:
                _, predicted = torch.max(outputs.data, 1)
            all_samples+= len(labels)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    n_correct+=1
                    results.append(explains[i])
                    picked_labels.append(labels[i])
                    picked_picked_outputs.append(outputs[i].detach().cpu().numpy())
        
        acc_at_top_k = []
        for result,label in progressbar.progressbar(zip(results,picked_labels),max_value=len(results)):
            imp = list(result[label+1])
            imp = [(w,i) for i,w in enumerate(imp)]
            imp = sorted(imp, key=operator.itemgetter(0), reverse=True)
            _,explains = determine_cls(result[0],class_priority,patterns,def_class)
            items = set()
            for pat in explains:
                for it in pat:
                    items.add(it)
            total = len(items)
            n_cor = 0
            for i in range(total):
                if imp[i][1] in items:
                    n_cor +=1
            acc_at_top_k.append(n_cor/total)
        in_results.append(np.mean(acc_at_top_k))
        print("seed: {} accuracy at top @ n is {} over {}".format(seed,np.mean(acc_at_top_k),len(acc_at_top_k)))
    print(name, "The accuracy at top @ n is {} over {}".format(np.mean(in_results),len(in_results)))
    
    
print("Interpretability evaluation on 2 classes:")

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


models = [LogisticRegression,FFNN,IFFNN,FFNN,FFNN,HWNet,HW_IFFNN,HWNet,HWNet]
model_names = ["LR","FC-BC1","FC-IFFNN-BC","FC-BC2","FC-BC3","HW-BC1","HW-IFFNN-BC","HW-BC2","HW-BC3"]
hyp_params = [(input_size,),(input_size, [hidden_size,hidden_size,hidden_size], num_classes, True),(input_size, [hidden_size,hidden_size,hidden_size], num_classes, True),(input_size, [int(1.3*hidden_size)]*3, num_classes, True),(input_size, [int(1.0*hidden_size)]*5, num_classes, True),(input_size, [hidden_size]*3, num_classes, True),(input_size, [hidden_size]*3, num_classes, True),(input_size, [int(1.105*hidden_size)]*3, num_classes, True),(input_size, [int(0.857*hidden_size)]*5, num_classes, True)]

setting_name = 'createddataset2cls'
    
train_loader,valid_loader,test_loader,patterns = get_data_loaders_created_dataset(batch_size,n_train,n_valid,n_test,n_dim,n_min_co,n_max_co,n_min_pattern,n_max_pattern,def_class,class_priority,pos_rate,seed)
device = torch.device("cuda:0")



bicls = True

root_res = './results/'
root_models = './models/'
seeds = [0,1,2,3,4]
for mod, name, hyp in zip(models,model_names,hyp_params):
    if "IFFNN" not in name and 'LR' not in name:
        continue
    print("name:",name)
    in_results = []
    for seed in seeds:
        need_reshape = False
        
        save_path = root_models+setting_name+"_"+name+"_seed"+str(seed)+"_model.pkl"
        net = mod(*hyp)
        checkpoint = torch.load(save_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
        net.eval()
        results = []
        picked_labels = []
        picked_picked_outputs = []
        all_samples = 0
        n_correct = 0
        for samples, labels in test_loader:
            samples = samples.to(device)
            if bicls:
                labels = labels.numpy()
            else:
                labels = labels.to(device)
            outputs = net(samples)
            explains = net.explain(samples)
            
            if bicls:
                predicted = np.round(F.sigmoid(outputs).detach().cpu().numpy())
            
            else:
                _, predicted = torch.max(outputs.data, 1)
            all_samples+= len(labels)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    n_correct+=1
                    results.append(explains[i])
                    picked_labels.append(labels[i])
                    picked_picked_outputs.append(outputs[i].detach().cpu().numpy())
        print("test accuracy:",n_correct/all_samples)
        #break
        
        
        acc_at_top_k = []
        for result,label in zip(results,picked_labels):
            imp = list(result[label+1])
            imp = [(w,i) for i,w in enumerate(imp)]
            imp = sorted(imp, key=operator.itemgetter(0), reverse=True)
            _,explains = determine_cls(result[0],class_priority,patterns,def_class)
            items = set()
            for pat in explains:
                for it in pat:
                    items.add(it)
            total = len(items)
            n_cor = 0
            for i in range(total):
                if imp[i][1] in items:
                    n_cor +=1
            acc_at_top_k.append(n_cor/total)
        in_results.append(np.mean(acc_at_top_k))
        print("seed: {} accuracy at top @ n is {} over {}".format(seed,np.mean(acc_at_top_k),len(acc_at_top_k)))
    print(name, "The accuracy at top @ n is {} over {}".format(np.mean(in_results),len(in_results)))
    
models = [SoftmaxRegression,FFNN,IFFNN,FFNN,FFNN,HWNet,HW_IFFNN,HWNet,HWNet]
model_names = ["SR","FC-MC1","FC-IFFNN-MC","FC-MC2","FC-MC3","HW-MC1","HW-IFFNN-MC","HW-MC2","HW-MC3"]
hyp_params = [(input_size, num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [int(2.73*hidden_size)]*3, num_classes),(input_size, [2*hidden_size]*5, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [int(1.682*hidden_size)]*3, num_classes),(input_size, [int(1.17*hidden_size)]*6, num_classes)]



bicls = False

root_res = './results/'
root_models = './models/'
seeds = [0,1,2,3,4]
for mod, name, hyp in zip(models,model_names,hyp_params):
    if "IFFNN" not in name and 'SR' not in name:
        continue
    print("name:",name)
    in_results = []
    for seed in seeds:
        need_reshape = False
        
        save_path = root_models+setting_name+"_"+name+"_seed"+str(seed)+"_model.pkl"
        net = mod(*hyp)
        checkpoint = torch.load(save_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
        net.eval()
        results = []
        picked_labels = []
        picked_picked_outputs = []
        all_samples = 0
        n_correct = 0
        for samples, labels in test_loader:
            samples = samples.to(device)
            if bicls:
                labels = labels.numpy()
            else:
                labels = labels.to(device)
            outputs = net(samples)
            explains = net.explain(samples)
            
            if bicls:
                predicted = np.round(outputs.detach().cpu().numpy())
            
            else:
                _, predicted = torch.max(outputs.data, 1)
            all_samples+= len(labels)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    n_correct+=1
                    results.append(explains[i])
                    picked_labels.append(labels[i])
                    picked_picked_outputs.append(outputs[i].detach().cpu().numpy())
        
        acc_at_top_k = []
        for result,label in progressbar.progressbar(zip(results,picked_labels),max_value=len(results)):
            imp = list(result[label+1])
            imp = [(w,i) for i,w in enumerate(imp)]
            imp = sorted(imp, key=operator.itemgetter(0), reverse=True)
            _,explains = determine_cls(result[0],class_priority,patterns,def_class)
            items = set()
            for pat in explains:
                for it in pat:
                    items.add(it)
            total = len(items)
            n_cor = 0
            for i in range(total):
                if imp[i][1] in items:
                    n_cor +=1
            acc_at_top_k.append(n_cor/total)
        in_results.append(np.mean(acc_at_top_k))
        print("seed: {} accuracy at top @ n is {} over {}".format(seed,np.mean(acc_at_top_k),len(acc_at_top_k)))
    print(name, "The accuracy at top @ n is {} over {}".format(np.mean(in_results),len(in_results)))
setting_name = 'createddataset2cls'