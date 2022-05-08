import torch
import numpy as np
import progressbar

def determine_cls(sam,class_priority,patterns,def_class):
    for cls in class_priority:
        reasons = []
        for pattern in patterns[cls]:
            if (sam[list(pattern)] == 1).all():
                reasons.append(pattern)
        if len(reasons) > 0:
            return cls, reasons
    return def_class, None
            
    
def generate_sample_with_pattern(pattern,cls,n_dim,class_priority,patterns,def_class,pos_rate):
    n_trial = 1
    sam = np.random.binomial(1, pos_rate, n_dim)
    sam[list(pattern)]=1
    real_cls,_ = determine_cls(sam,class_priority,patterns,def_class)
    while real_cls != cls:
        sam = np.random.binomial(1, pos_rate, n_dim)
        sam[list(pattern)]=1
        real_cls,_ = determine_cls(sam,class_priority,patterns,def_class)
        n_trial += 1
    return tuple(sam), n_trial    
    

def gen_INBEN(n_train, n_valid, n_test, n_dim, n_min_co, n_max_co, n_min_pattern, n_max_pattern, def_class, class_priority, pos_rate=0.2, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    n_class = len(class_priority)
    assert(set(class_priority)==set(range(n_class)))
    assert(n_train%n_class==0)
    assert(n_valid%n_class==0)
    assert(n_test%n_class==0)
    
    patterns = {}
    features = np.array(list(range(n_dim)))
    all_patterns = {}
    for i in range(n_class):
        n_pattern = np.random.randint(n_min_pattern, n_max_pattern+1)
        print("class {} has {} patterns".format(i,n_pattern))
        patterns[i]=[]
        for j in range(n_pattern):
            pattern_length = np.random.randint(n_min_co, n_max_co+1)
            pattern = tuple(np.random.choice(features,pattern_length,False))
            while pattern in all_patterns:
                pattern_length = np.random.randint(n_min_co, n_max_co+1)
                pattern = tuple(np.random.choice(features,pattern_length,False))
            all_patterns[pattern] = i
            patterns[i].append(pattern)
            
    n_sam_per_class = int((n_train+n_valid+n_test)/n_class)
    
    all_samples = set()
    n_trials = []
    n_repeats = []
    samples_per_class = {}
    for i in range(n_class):
        print("generating class:",i)
        samples_per_class[i] = []
        n_pattern = len(patterns[i])
        bar = progressbar.ProgressBar(max_value=n_sam_per_class)
        for j in range(n_pattern):
            pattern = patterns[i][j]
            n_repeat = 0
            sam, n_trial = generate_sample_with_pattern(pattern,i,n_dim,class_priority,patterns,def_class,pos_rate)
            n_trials.append(n_trial)
            while sam in all_samples:
                sam, n_trial = generate_sample_with_pattern(pattern,i,n_dim,class_priority,patterns,def_class,pos_rate)
                n_trials.append(n_trial)
                n_repeat += 1
            n_repeats.append(n_repeat)
            samples_per_class[i].append(list(sam))
            all_samples.add(sam)
            bar.update(len(samples_per_class[i]))
        while len(samples_per_class[i]) < n_sam_per_class:
            bar.update(len(samples_per_class[i]))
            pattern = np.random.choice(patterns[i],1)
            n_repeat = 0
            sam, n_trial = generate_sample_with_pattern(pattern,i,n_dim,class_priority,patterns,def_class,pos_rate)
            n_trials.append(n_trial)
            while sam in all_samples:
                sam, n_trial = generate_sample_with_pattern(pattern,i,n_dim,class_priority,patterns,def_class,pos_rate)
                n_trials.append(n_trial)
                n_repeat += 1
            n_repeats.append(n_repeat)
            samples_per_class[i].append(list(sam))
            all_samples.add(sam)

    train_set = []
    valid_set = []
    test_set = []
    
    n_train_per_class = int(n_train/n_class)
    n_valid_per_class = int(n_valid/n_class)
    n_test_per_class = int(n_test/n_class)
    
    for i in range(n_class):
        train_set.extend([(torch.tensor(sam,dtype=torch.float32),i) for sam in samples_per_class[i][:n_train_per_class]])
        valid_set.extend([(torch.tensor(sam,dtype=torch.float32),i) for sam in samples_per_class[i][n_train_per_class:n_train_per_class+n_valid_per_class]])
        test_set.extend([(torch.tensor(sam,dtype=torch.float32),i) for sam in samples_per_class[i][n_train_per_class+n_valid_per_class:]])

    np.random.shuffle(train_set)
    np.random.shuffle(valid_set)
    np.random.shuffle(test_set)
    print("avg trial:",np.mean(n_trials))
    print("avg repeats:",np.mean(n_repeats))
    
    return train_set, valid_set, test_set, patterns






























    

    