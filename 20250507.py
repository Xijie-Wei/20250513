# This code is about to train the model with triplet loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
from numpy import genfromtxt
from torch.utils.data import DataLoader,TensorDataset
from training_function import test_score
from tqdm import tqdm

# load data
folder = "75_dataset_nhit_eta0_theta_s"
train_input = torch.load(f"data_eta_test/{folder}/data_train_input.csv",weights_only=False)
train_label = torch.load(f"data_eta_test/{folder}/data_train_label.csv",weights_only=False)
print(train_input.shape)
print(train_label.shape)


# Create triplet loss data loader
class triple_set(Dataset):

    def __init__(self,inputs,labels,dataset_size = 50000):

        self.inputs = inputs
        self.good_tracks_id = ((labels[:,0] == 1).nonzero(as_tuple=True)[0])
        self.bad_tracks_id = ((labels[:,1] == 1).nonzero(as_tuple=True)[0])
        self.num_triplets = dataset_size

        #print(self.good_tracks_id.shape)
        #print(self.bad_tracks_id.shape)

        self.triplets = self.__generate_triplets__()

    def __hard_mining__(self,anchor_index,dim_num = 3):
        '''
         This function is about find the closing negative with anchor

         Inputs:
            anchor_index: index of anchor
            dim_num: dimention of inputs, defult 3

         Outputs: 
            idx_nag: index of self.good_tracks_id with the point of furthest positive
            idx_nag: index of self.bad_tracks_id wiith the point of cloest nagitve
        '''
        anchor = self.inputs[anchor_index].to('cuda')
        distance_close = torch.zeros(self.bad_tracks_id.shape[0]).to('cuda')
        distance_far = torch.zeros(self.good_tracks_id.shape[0]).to('cuda')
        self.inputs = self.inputs.to('cuda')
        for dim in range (dim_num):
            distance_close = torch.add(distance_close,torch.square(self.inputs[self.bad_tracks_id,dim]-anchor[dim]))
            distance_far = torch.add(distance_far,torch.square(self.inputs[self.good_tracks_id,dim]-anchor[dim]))
        #print(distance_close)
        idx_nag = torch.argmin(distance_close)
        idx_pos = torch.argmax(distance_far)
        self.inputs = self.inputs.cpu()
        idx_pos,idx_nag = idx_pos.cpu(),idx_nag.cpu()
        return idx_pos,idx_nag


    def __generate_triplets__(self):
        triplets_index = []
        for i in tqdm(range(self.num_triplets)):
            anchor_index_temp = np.random.randint(0,self.good_tracks_id.shape[0])
            good_index_left = self.good_tracks_id[self.good_tracks_id!=anchor_index_temp].clone()
            #positive_index_temp = np.random.randint(0,good_index_left.shape[0])
            #negative_index_temp = np.random.randint(0,self.bad_tracks_id.shape[0])
            positive_index_temp,negative_index_temp = self.__hard_mining__(anchor_index_temp,dim_num = 3)
            triplets_index.append((self.good_tracks_id[anchor_index_temp],good_index_left[positive_index_temp],self.bad_tracks_id[negative_index_temp]))
        return triplets_index
    
    # port for class dataset
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self,idx):
        anc,pos,neg = self.triplets[idx]
        anchor = self.inputs[anc]
        postive = self.inputs[pos]
        negtive= self.inputs[neg]
        return anchor,postive,negtive
    

# Create dataset
trainset_size = 250000
Shuffling = False
if not(Shuffling):
    train_set = triple_set(train_input,train_label,dataset_size=trainset_size)
    print("Training set created")
    train_data = DataLoader(dataset=train_set,
                            batch_size = 1,
                            num_workers = 0,
                            shuffle = True,
                            pin_memory = True)

# Declare a network
class net(nn.Module):

    def __init__(self):
        super().__init__()

        self.classi = nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Linear(64,2)
        )
    def forward(self,inputs):
        output = self.classi(inputs)

        return output

network = net()
network.to('cuda')
print(net)

# Load pre_trained model( if needed)
Load_model = False
if Load_model:
    PATH = './triplet_network.pth'
    network = net()
    network.load_state_dict(torch.load(PATH, weights_only=True))
    network.eval()
    network.to('cuda')

# training loop
triplet_loss = nn.TripletMarginLoss(margin=1)
optimizer = optim.SGD(network.parameters(),lr=0.0001)

print("Start training")
total_enpoch = 10
for epoch in tqdm(range (total_enpoch)):
    total_loss = 0
    tqdm.write("----------------------------------------")
    tqdm.write(f"Epoch: {epoch+1}")
    #print(f"Progress: {100*epoch/total_enpoch:3f}%")
    if Shuffling:
        # Create dataset
        tqdm.write("Creating training set")
        train_set = triple_set(train_input,train_label,dataset_size=trainset_size)
        tqdm.write("Training set created")
        train_data = DataLoader(dataset=train_set,
                                batch_size = 1,
                                num_workers = 0,
                                shuffle = True,
                                pin_memory = True)

    tqdm.write("Epoch start")
    for anchoc,posstive,negative in tqdm(train_data):

        anchoc,posstive,negative = anchoc.to('cuda'),posstive.to('cuda'),negative.to('cuda')
        anch = network(anchoc)
        poss = network(posstive)
        nega = network(negative)

        optimizer.zero_grad()
        loss = triplet_loss(anch,poss,nega)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    tqdm.write("Epoch finished")
    tqdm.write(f"Loss: {total_loss / trainset_size:3f}")

#save_model 
PATH = './triplet_network.pth'
torch.save(network.state_dict(), PATH)
print("Model saved")