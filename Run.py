import os
import torch
import torch.nn as neural_net
import pandas as pd
import numpy as np
import random

def load_file(file_name):
    df = pd.read_csv(file_name)
    user = df["users"]
    item = df["items"]

    actual_dict = {}
    for user, sf in df.groupby("users"):
        actual_dict[user] = list(sf["items"])

    data = df[["users", "items"]].to_numpy(dtype=int).tolist()
    return data, actual_dict, set(df["users"]), set(df["items"])

class Dataset:

    def __init__(self, path, name):
        if name == "books":
            self.train_path = os.path.join(path, "books_train.csv")
            self.test_path = os.path.join(path, "books_test.csv")
        
        elif name == "movies":
            self.train_path = os.path.join(path, "movies_train.csv")
            self.test_path = os.path.join(path, "movies_test.csv")

        
        self.initialize()
    
    def initialize(self):
        self.train_data, self.train_dict, train_user_set, train_item_set = load_file(self.train_path)
        self.test_data, self.test_dict, test_user_set, test_item_set = load_file(self.test_path)

        assert (test_user_set.issubset(train_user_set))
        assert (test_item_set.issubset(train_item_set))
        self.user_set = train_user_set
        self.item_set = train_user_set
        self.num_user = len(train_user_set)
        self.num_item = len(train_item_set)
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)

        self.test_input_dict, self.train_neg_dict = self.get_dicts()
        
        
        print("Train size:", self.train_size)
        print("Test size:", self.test_size)
        print("Number of user:", self.num_user)
        print("Number of item:", self.num_item)
        print("Data Sparsity: {:.1f}%".format(100 * (self.num_user * self.num_item - self.train_size)/ (self.num_user * self.num_item)))

        print()
    
    def get_dicts(self):
        train_actual_dict, test_actual_dict = self.train_dict, self.test_dict
        train_neg_dict = {}
        test_input_dict = {}
        for user in list(self.user_set):
            train_neg_dict[user] = list(self.item_set - set(train_actual_dict[user]))

        for user in test_actual_dict.keys():
            test_input_dict[user] = train_neg_dict[user]
            train_neg_dict[user] = list(set(train_neg_dict[user]) - set(test_actual_dict[user]))
    
        return test_input_dict, train_neg_dict

    def neg_sampling(self, num):
        item_dict = self.train_neg_dict
        user_list = []
        item_list = []

        for user in list(self.user_set):
            items = random.sample(item_dict[user], 20)
            item_list += items
            user_list += [user] * len(items)
        result = np.transpose(np.array([user_list, item_list]))
        return random.sample(result.tolist(), num)
    
    def get_train(self):
        neg = self.neg_sampling(num=self.train_size)
        pos = self.train_data
        labels = [1] * len(pos) + [0] * len(neg)
        return pos+neg, labels
    
    def get_data(self):
        return self.test_dict, self.test_input_dict, self.test_data, self.num_user, self.num_item


def training(train_data_1, train_data_2, label_data_1, label_data_2):
    df_data_1 = pd.DataFrame(train_data_1, columns=["users_data_1", "items_data_1"])
    df_data_2 = pd.DataFrame(train_data_2, columns=["users_data_2", "items_data_2"])
    df_data_1["label_data_1"] = label_data_1
    df_data_2["label_data_2"] = label_data_2
    train_frames = []
    for user, sf_data_1 in df_data_1.groupby("users_data_1"):
        df = pd.DataFrame()
        sf_data_2 = df_data_2.loc[df_data_2["users_data_2"] == user]
        l = min(len(sf_data_1), len(sf_data_2))
        
        df["users"] = l*[user]
       
        sample_data_1 = sf_data_1.sample(n=l)
      
        sample_data_2 = sf_data_2.sample(n=l)

        df["items_data_1"] = sample_data_1["items_data_1"].values
        df["items_data_2"] = sample_data_2["items_data_2"].values
        df["labels_data_1"] = sample_data_1["label_data_1"].values
        df["labels_data_2"] = sample_data_2["label_data_2"].values
        train_frames.append(df)
    frame = pd.concat(train_frames)
    data = frame[["users", "items_data_1", "items_data_2"]].values.tolist()
    labels = frame[["labels_data_1", "labels_data_2"]].values.tolist()
    return data, labels



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CoNet_Model(neural_net.Module):

    def __init__(self):
        super(CoNet_Model, self).__init__()
        
        dataset_path = os.path.join(os.path.join(os.getcwd(), '../../dataset'),"books_and_movies")
        print("--------------------------Books---------------------------")
        self.dataset_data_1 = Dataset(dataset_path, "movies")
        
        print("--------------------------Movies--------------------------")
        self.dataset_data_2 = Dataset(dataset_path, "books")
      
        self.test_dict_data_1, self.test_input_dict_data_1, self.test_data_data_1, self.num_user_data_1, self.num_item_data_1 = self.dataset_data_1.get_data()
        self.test_dict_data_2, self.test_input_dict_data_2, self.test_data_data_2, self.num_user_data_2, self.num_item_data_2 = self.dataset_data_2.get_data()

        self.reg = 0.0001
        self.batch_size = 32
        self.cross_layer = 2

        self.edim = 32
        self.lr = 0.001

        self.std =  0.01
        self.initialise_neural_net()

  
        self.U = neural_net.Embedding(self.num_user_data_1, self.edim)
        self.V_data_1 = neural_net.Embedding(self.num_item_data_1, self.edim)
        self.V_data_2 = neural_net.Embedding(self.num_item_data_2, self.edim)
        self.sigmoid = neural_net.Sigmoid()
    
    def create_neural_net(self, layers):
        weights = {}
        biases = {}
        for l in range(len(self.layers) - 1):
            weights[l] = torch.normal(mean=0, std=self.std, size=(layers[l], layers[l+1]), requires_grad=True, device=device)
            biases[l] = torch.normal(mean=0, std=self.std, size=(layers[l+1],), requires_grad=True, device=device)
        return weights, biases
    
    def initialise_neural_net(self):
        edim = 2*self.edim
        i=0
        self.layers = [edim]
        while edim>8:
            i+=1
            edim/=2
            self.layers.append(int(edim))

        assert (self.cross_layer <= i)
 
        
        self.weights_biases_data_1()
        self.weights_biases_data_2()

       
        weights_shared = {}
        for l in range(self.cross_layer):
            weights_shared[l] = torch.normal(mean=0, std=self.std, size=(self.layers[l], self.layers[l+1]), requires_grad=True, device=device)
        self.weights_shared = [weights_shared[i] for i in range(self.cross_layer)]

    def weights_biases_data_1(self) :

        weights_data_1, biases_data_1 = self.create_neural_net(self.layers)
        self.weights_data_1 = neural_net.ParameterList([neural_net.Parameter(weights_data_1[i]) for i in range(len(self.layers) - 1)])
        self.biases_data_1 = neural_net.ParameterList([neural_net.Parameter(biases_data_1[i]) for i in range(len(self.layers) - 1)])
        self.W_data_1 = neural_net.Parameter(torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True, device=device))
        self.b_data_1 = neural_net.Parameter(torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True, device=device))

    def weights_biases_data_2(self) : 
      
        weights_data_2, biases_data_2 = self.create_neural_net(self.layers)
        self.weights_data_2 = neural_net.ParameterList([neural_net.Parameter(weights_data_2[i]) for i in range(len(self.layers) - 1)])
        self.biases_data_2 = neural_net.ParameterList([neural_net.Parameter(biases_data_2[i]) for i in range(len(self.layers) - 1)])
        self.W_data_2 = neural_net.Parameter(torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True, device=device))
        self.b_data_2 = neural_net.Parameter(torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True, device=device))


        
    def multi_layer_feedforward(self, user, item_data_1, item_data_2):
        user_emb = self.U(torch.LongTensor(user)).to(device=device)
        item_emb_data_1 = self.V_data_1(torch.LongTensor(item_data_1)).to(device=device)
        item_emb_data_2 = self.V_data_2(torch.LongTensor(item_data_2)).to(device=device)

        cur_data_1 = torch.cat((user_emb, item_emb_data_1), 1)
        cur_data_2 = torch.cat((user_emb, item_emb_data_2), 1)
        pre_data_1 = cur_data_1
        pre_data_2 = cur_data_2
        for l in range(len(self.layers) - 1):
            
            cur_data_1 = torch.add(torch.matmul(cur_data_1, self.weights_data_1[l]), self.biases_data_1[l])
            cur_data_2 = torch.add(torch.matmul(cur_data_2, self.weights_data_2[l]), self.biases_data_2[l])

            if (l < self.cross_layer):
              
                cur_data_1 = torch.matmul(pre_data_2, self.weights_shared[l])
                cur_data_2 = torch.matmul(pre_data_1, self.weights_shared[l])
            cur_data_1 = neural_net.functional.relu(cur_data_1)
            cur_data_2 = neural_net.functional.relu(cur_data_2)
            pre_data_1 = cur_data_1
            pre_data_2 = cur_data_2

        z_data_1 = torch.matmul(cur_data_1, self.W_data_1) + self.b_data_1
        z_data_2 = torch.matmul(cur_data_2, self.W_data_2) + self.b_data_2
       
        return self.sigmoid(z_data_1), self.sigmoid(z_data_2)
    
    def model_fit(self, num_epoch=101):
        
        params = [{"params": self.parameters(), "lr":0.001},
                  {"params": self.weights_shared, "lr": 0.001, "weight_decay":0.0001}]
        optimizer = torch.optim.Adam(params)
        criterion = neural_net.MSELoss()

        data_data_1, labels_data_1 = self.dataset_data_1.get_train()
        data_data_2, labels_data_2 = self.dataset_data_2.get_train()
        data, labels = training(data_data_1, data_data_2, labels_data_1, labels_data_2)

        train_data = torch.tensor(data)
        labels = torch.tensor(labels, device=device)
        print(labels.shape)
        labels_data_1, labels_data_2 = labels[:,0], labels[:,1]
        user, item_data_1, item_data_2 = train_data[:,0], train_data[:,1], train_data[:,2]
        for epoch in range(num_epoch):
            permut = torch.randperm(user.shape[0])
            max_idx = int((len(permut) // (16) -1) * (16))
            range(0, max_idx, 32)
            for batch in range(0, max_idx, 32):
                optimizer.zero_grad()
                idx = permut[batch : batch + 32]  
                pred_data_1, pred_data_2 = self.multi_layer_feedforward(user[idx], item_data_1[idx], item_data_2[idx])
                training_loss_data_1 = criterion(labels_data_1[idx].float(), torch.squeeze(pred_data_1))
                training_loss_data_2 = criterion(labels_data_2[idx].float(), torch.squeeze(pred_data_2))
                training_loss = training_loss_data_1 + training_loss_data_2
                training_loss.backward()
                optimizer.step()
            if epoch % 5 == 0:    
                print("epoch {} training loss : {:.4f}".format(epoch, training_loss))

 
CoNet_Model().model_fit()
