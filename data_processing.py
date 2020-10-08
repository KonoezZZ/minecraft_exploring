##### load source data #####
import numpy as np
import minerl

data_dir = r'\content\gdrive\My Drive\minecraft\dataset'
stream_name = "MineRLNavigateExtremeDense-v0"   # or "MineRLNavigateDense-v0"
data = minerl.data.make(stream_name, data_dir, force_download=False)
data_names = np.sort(data.get_trajectory_names())   # 154 samples  

##### Customize dataset #####
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CustomizedDataset:
    def __init__(self,
          data,
          train_size=0.8,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          zero_camera=True,
          zero_jump=False,
          num_tasks=10):
        self.data = data
        self.train_size = train_size
        self.zero_camera = zero_camera
        self.num_tasks = num_tasks
        self.zero_jump = zero_jump
        if self.data is not None:
          self.customize()
          self.data_names = self.data.get_trajectory_names()
    
    def customize(self): # ~3mins; 276917 samples
        states = []
        actions = []
        camera_actions = []
        next_states = []
        for task_num in range(self.num_tasks): #range(len(self.data_names)-1):
            for sample in self.data.load_data(data_names[task_num]):
                if sample[4] == 1:
                    # discard terminal data
                    continue
                if sample[1]['camera'][0] != 0 or sample[1]['camera'][1] != 0:
                    if self.zero_camera == True:
                        continue
                    else:
                      pass        
                if sample[1]['jump'] != 0:
                    if self.zero_jump == True:
                        continue
                    else:
                      pass
                if  sample[1]['forward']==0 and sample[1]['back']==0 and sample[1]['left']==0 and sample[1]['right']==0:
                    pass
                states.append(sample[0]['pov'])
                actions.append(np.array([
                      sample[1]['forward'] - sample[1]['back'],     # 1 forward; -1 backward; 0 neither
                      sample[1]['right'] - sample[1]['left'],       # 1 right; -1 left; 0 neither
                      sample[1]['jump']                             # 1 jump; 0 not jump
                ]).astype(int))
                camera_actions.append(
                      int(sample[1]['camera'][0] > 0) - int(sample[1]['camera'][0] < 0)
                )
                next_states.append(sample[3]['pov'])            
        self.length = len(actions)
    
        # adjust axes and transform to tensors
        self._states = torch.tensor(states, device=device).permute(0,3,1,2)  #BETTER TO USE NP.MOVEAXIS: MUCH FASTER
        self._actions = torch.tensor(actions, device=device).unsqueeze(1)
        self._camera_actions = torch.tensor(camera_actions, device=device).unsqueeze(1)
        self._next_states = torch.tensor(next_states, device=device).permute(0,3,1,2)
    
    def train_valid_split(self, num_samples=None, gap=0):
        if num_samples == None:
            num_samples = self.length
        train_size = int(num_samples * self.train_size)
        train = [self._states[:num_samples-gap][:train_size],
              self._actions[:num_samples-gap][:train_size],
              self._camera_actions[:num_samples-gap][:train_size],
              self._next_states[gap:num_samples][:train_size]
            ]
        valid = [self._states[:num_samples-gap][train_size:],
              self._actions[:num_samples-gap][train_size:],
              self._camera_actions[:num_samples-gap][train_size:],
              self._next_states[gap:num_samples][train_size:]
            ]
        return train, valid
    
    def load(self, pt):
        self._states = pt[0]
        self._actions = pt[1]
        self._camera_actions = pt[2]
        self._next_states = pt[3]
        self.length = len(self._actions)
    
    @property
    def len(self):
        return self.length
    
    @property
    def states(self):
        return self._states #(num_samples,3,64,64)=(N,C,H,W)
    
    @property
    def actions(self):
        return self._actions #(num_samples,1,3)=(N,C,L)
    
    @property
    def camera_actions(self):
        return self._camera_actions #(num_samples,1,2)
    
    @property
    def next_states(self):
        return self._next_states #(num_samples,3,64,64)
    
    

class MakeDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


    
class MakeAutoencoderDataset(Dataset):
    
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index]

    
      
def get_loaders(train, valid, MODE, BATCH_SIZE, shuffle=True, num_workers=0):   # non-zero num_workers will get bugs in GPU mode
    if MODE == 's2a':
        x_train, y_train, x_valid, y_valid = train[0], train[1], valid[0], valid[1]
        train_set = MakeDataset(x_train, y_train)
        valid_set = MakeDataset(x_valid, y_valid)
        
    elif MODE == 's2camera':
        x_train, y_train, x_valid, y_valid = train[0], train[2], valid[0], valid[2]
        train_set = MakeDataset(x_train, y_train)
        valid_set = MakeDataset(x_valid, y_valid)

    elif MODE == 's2s':
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        x_train, y_train, x_valid, y_valid = train[0].type(dtype), train[3].type(dtype), valid[0].type(dtype), valid[3].type(dtype)
        train_set = MakeDataset(x_train/255, y_train/255)
        valid_set = MakeDataset(x_valid/255, y_valid/255)
    
    elif MODE == 'sa2s':
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        x_train = torch.cat(
                        (train[0].flatten(start_dim=1).type(dtype),
                         train[1].flatten(start_dim=1).type(dtype)),
                        dim=1)    # ( BATCH_SIZE, 3*64*64*num_available_actions )
        y_train = train[3]
        x_valid = torch.cat(
                        (valid[0].flatten(start_dim=1).type(dtype),
                         valid[1].flatten(start_dim=1).type(dtype)),
                        dim=1) 
        y_valid = valid[3]
        train_set = MakeDataset(x_train, y_train)
        valid_set = MakeDataset(x_valid, y_valid)
        
    elif MODE == 'autoencoder':
        x_train, x_valid = train[0], valid[0]
        train_set = MakeAutoencoderDataset(x_train)
        valid_set = MakeAutoencoderDataset(x_valid)
        
    
    else:
        print("ERROR: MODE NOT FOUND")

        
    train_loader = DataLoader(train_set,
                  batch_size=BATCH_SIZE,
                  shuffle=shuffle,
                  num_workers=num_workers)  
        
    valid_loader = DataLoader(valid_set,
                  batch_size=BATCH_SIZE,
                  shuffle=False,
                  num_workers=num_workers)    
        
    return train_loader, valid_loader 
    
##### save dataloaders #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CustomizedDataset(data=data, train_size=0.8, device=device, zero_camera=False, zero_jump=False, num_tasks=60)
torch.save(dataset.train_valid_split()[0], '/content/gdrive/My Drive/minecraft/customized_data/Navigate_train.pt')
torch.save(dataset.train_valid_split()[1], '/content/gdrive/My Drive/minecraft/customized_data/Navigate_valid.pt')
