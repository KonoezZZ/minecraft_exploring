from . import prerequisites
from . import data_processing
from . import s2a
from . import s2s

def load_sa2s(action_class):
    model = UNet_sa2s(3,
        depth=3,
        start_filts=8,
        merge_mode='concat')
    model.load_state_dict(torch.load('SA2S_MODEL_NAME'+str(action_class)+'.pt'))
    return model


def load_all_sa2s():
    models = []
    for i in range(len(available_actions)):
      models.append(load_sa2s(i))
    return models

def load_s2s():
    model = UNet_s2s(3,
        depth=4,
        start_filts=8,
        merge_mode='concat')
    model.load_state_dict(torch.load('S2S_MODEL_NAME.pt'))
    return model


def get_actions(state, s2s, sa2s, multipliers=np.ones(len(available_actions))):
    X = state
    y = s2s(X)
    # iterate sa2s for each action
    best_a = 0
    losses = np.ones([len(y), len(sa2s)]) * 9999   # batch行，action列
    for a, model in enumerate(sa2s):
        pred_y = model(X)
        for i in range(len(y)):
            losses[i, a] = F.mse_loss(pred_y[i], y[i]).item()
    losses = np.einsum('ij, j -> ij', losses, multipliers)
    best_actions = np.argmin(losses, axis=1)
    return best_actions


def action2num(y):
    act_val = y.squeeze().numpy()
    act_val_classes = np.full((len(act_val)), -1, dtype=np.int)
    for i, a in enumerate(available_actions):
        act_val_classes[np.all(act_val == a, axis=1)] = i
    return act_val_classes


def comb_test(multipliers=np.ones(len(available_actions))):
    sa2s = load_all_sa2s()
    s2s = load_s2s()
    train_loader, valid_loader = get_loaders(train_set, valid_set, MODE='s2a', BATCH_SIZE=32)
    acc = []
    count = 0
    action_counts = np.zeros(len(available_actions), dtype=np.int)
    for i, data in enumerate(valid_loader):
        X, y = data[0].float()/255, data[1]
        pred_actions = get_actions(X, s2s, sa2s, multipliers)
        actions = action2num(y)
        for a in actions:
          action_counts[a] += 1
        acc.append(sum(pred_actions==actions)/len(actions))
        count += 1
        if count == 100:
            break
    return sum(acc)/len(acc), action_counts
    
  
 
import time

multipliers_list = [
             #np.ones(len(available_actions)),
             np.array([1.5, 1, 1, 1, 1, 1, 1.1, 1.1, 1.1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
             #np.array([1.2, 0.8, 0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8]),
             #np.array([1.4, 0.7, 0.7, 0.7, 0.7, 0.7, 1.4, 1.4, 1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7, 0.7])
]

for multipliers in multipliers_list:
    start = time.time()
    acc, acts = comb_test(multipliers)
    print('------------------------------------------------')
    print('Weights: {}, Accuracy: {}'.format(multipliers, acc))
    print('Action_counts: {}, Time spent: {}'.format(acts, time.time()-start))
