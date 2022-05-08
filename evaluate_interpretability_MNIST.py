from model import *
import matplotlib.pyplot as plt 


input_size = 784 
hidden_size = 500 
num_classes = 10
batch_size = 256
num_epochs = 200
learning_rate = 0.001 


def show_interpretation_image(mod, name, hyp, n_sam):
    print("name:",name)    
    print("mod:",mod)    
    if "CNN" in name or "ResNET" in name:
        print("model:",name,"doesn't need reshape")
        need_reshape = False
    else:
        print("model:",name,"needs reshape")
        need_reshape = True
    save_path = root_models+setting_name+"_"+name+"_seed0_model.pkl"
    net = mod(*hyp)
    checkpoint = torch.load(save_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(device)
    net.eval()
    for images, labels in test_loader:
        if need_reshape:
            images = images.reshape(-1, images.shape[1]*images.shape[2]*images.shape[3])
        images = images.to(device)
        if bicls:
            labels = labels.numpy()
        else:
            labels = labels.to(device)
        outputs = net(images)
        explains = net.explain(images)
        
        if bicls:
            predicted = np.round(F.sigmoid(outputs).detach().cpu().numpy())
        
        else:
            _, predicted = torch.max(outputs.data, 1)
        results = []
        picked_labels = []
        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                results.append(explains[i])
                picked_labels.append(labels[i])
        break

    for i in range(n_sam):
        ax = plt.subplot(n_sam,num_classes+1,(1+num_classes)*i+1)
        plt.imshow(results[i][0].reshape((28,28)), cmap='gray')
        if i == 0:
            ax.set_title("Original")
        plt.axis('off')
        for j in range(num_classes):
            ax = plt.subplot(n_sam,num_classes+1,(1+num_classes)*i+2+j)
            plt.imshow(results[i][j+1].reshape((28,28)), cmap='gray')
            plt.axis('off')
            if i == 0:
                ax.set_title(str(j))
    plt.show() 
    

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
device = torch.device("cuda:0")

bicls = True

root_res = './results/'
root_models = './models/'

n_sam = 5
inds = [0,2,6,10,14]
for ind in inds:
    show_interpretation_image(models[ind],model_names[ind],hyp_params[ind], n_sam)
    
    
    

root_res = './results/'
root_models = './models/'
all_classes =  [0,1]
num_classes = len(all_classes)
models = [SoftmaxRegression,FFNN,IFFNN,FFNN, FFNN, CNNNet, I_CNNNet, CNNNet2_MC2, CNNNet2_MC3,ResNet,ResNet_IFFNN,ResNet2_MC2,ResNet2_MC3,HWNet,HW_IFFNN,HWNet,HWNet]
model_names = ["SR","FC-MC1","FC-IFFNN-MC","FC-MC2","FC-MC3","CNN-MC1","CNN-IFFNN-MC","CNN-MC2","CNN-MC3","ResNET-MC1","ResNET-IFFNN-MC","ResNET-MC2","ResNET-MC3","HW-MC1","HW-IFFNN-MC","HW-MC2","HW-MC3"]
hyp_params = [(input_size, num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [hidden_size,hidden_size,hidden_size], num_classes),(input_size, [int(1.481*hidden_size)]*3, num_classes),(input_size, [int(1.11*hidden_size)]*5, num_classes),(num_classes,),(num_classes,),(num_classes,),(num_classes,),(ResidualBlock,[2, 2, 2, 2],num_classes),(ResidualBlock,[2, 2, 2, 2],num_classes),(ResidualBlock,[2, 2, 2, 2],num_classes),(ResidualBlock,[2, 2, 2, 2],num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [hidden_size]*3, num_classes),(input_size, [int(1.167*hidden_size)]*3, num_classes),(input_size, [449]*5, num_classes)]

seeds = [0,1,2,3,4]
setting_name = '2cls01'
assert(len(models)==len(model_names))
assert(len(hyp_params)==len(model_names))

bicls = False


n_sam = 5
inds = [0,2,6,10,14]
for ind in inds:
    show_interpretation_image(models[ind],model_names[ind],hyp_params[ind], n_sam)