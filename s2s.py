from . import prerequisites
from . import data_processing



##### load your data #####
train_set = torch.load(r'YOUR_PATH/LOADER_NAME_train.pt')  
valid_set = torch.load(r'YOUR_PATH/LOADER_NAME_valid')  
