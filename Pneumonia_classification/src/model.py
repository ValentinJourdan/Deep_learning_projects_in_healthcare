import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.nn.parallel import DataParallel
from .utils import build
from .network import Net



class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.iter = 0 if 'iter' not in opt else opt['iter']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == 'gpu':
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs")
                self.data_parallel()
        self.network_opt = opt['networks'][0]
        net = build(Net, self.network_opt['args'])
        if 'path' in self.network_opt.keys():
            self.load_net(net, self.network_opt['path'])
        self.network = net


        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.network.parameters(), lr=opt['lr'], momentum=opt['momentum'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt['num_iters'], eta_min=0.00001)
        self.num_folds = opt['num_folds']
        self.num_epochs = opt['num_iters']
        self.batch_size = opt['batch_size']

        self.best_score = None
        self.save_best_only = True 
        self.print_every = self.opt['print_every']

    def train(self, model, train_images, train_labels):
        self.network.train()    
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for fold, (train_index, val_index) in enumerate(kf.split(train_images)):
            print(f'Fold {fold + 1}/{self.num_folds}')
            
            # Split data
            train_images_fold = train_images[train_index]
            train_labels_fold = train_labels[train_index]
            val_images_fold = train_images[val_index]
            val_labels_fold = train_labels[val_index]

            
            for epoch in range(self.num_epochs):
                running_loss = 0.0
                running_correct = 0
                dataloader = tqdm(range(0, len(train_images_fold), self.batch_size), position=0, leave=True)

                for i in dataloader:
                    inputs = train_images_fold[i:i+self.batch_size]
                    labels = train_labels_fold[i:i+self.batch_size]


                    self.optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs,labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    predicted = (outputs>0.5).int()
                    running_correct += (predicted == labels).int().sum()

                    dataloader.set_description(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(train_images_fold):.6f}, Accuracy : {100*running_correct/len(train_images_fold):.1f}%')
                    dataloader.refresh()

            dataloader.close()

            # Validation phase
            self.network.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for i in range(0, len(val_images_fold), self.batch_size):
                    inputs = val_images_fold[i:i+self.batch_size]
                    labels = val_labels_fold[i:i+self.batch_size]

                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    predicted = (outputs>0.5).int()
                    val_correct += (predicted == labels).int().sum()
            
            avg_loss = val_loss / len(val_images_fold)
            if not self.save_best_only or self.best_score is None or avg_loss < self.best_score:
                # Update the best score and save the model
                self.best_score = avg_loss
                self.save_model()
                    
            print(f'Fold {fold + 1}/{self.num_folds} - Validation Loss: {val_loss/len(val_images_fold):.5f} - Validation Accuracy : {100*val_correct/len(val_images_fold):.1f}%')

        print('Cross-validation finished')
    
    def load_net(self, net, path):
        state_dict = torch.load(path, map_location=self.device)
        net.load_state_dict(state_dict)

    def load_model(self, path):
        load_dict = torch.load(path, map_location = self.device)
        self.iter = load_dict.get('iter', 0) # if 'iter' is not in the dict the value will be 0 
        self.optimizer.load_state_dict(load_dict['optimizer'])
        self.scheduler.load_state_dict(load_dict['scheduler'])
        self.network.load_state_dict(load_dict['net'])
    
    def save_net(self):
        net = self.network
        if isinstance(net, DataParallel):
            net = net.module
        torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], f'{self.network_opt['name']}_ft.pth'))

    def save_model(self):
        save_dict = {'iter': self.iter,
                    'optimizer_BNN': self.optimizer.state_dict(),
                    'scheduler_BNN': self.scheduler.state_dict(),
                    'net': self.network.state_dict()}
        torch.save(save_dict, os.path.join(self.opt['log_dir'], f'{self.network_opt['name']}.pth'))
        print(f"Model saved to {os.path.join(self.opt['log_dir'], f'{self.network_opt['name']}.pth')} \n")

    def data_parallel(self):
        net = self.network
        net = net.cuda()
        net = DataParallel(net)
        self.network = net
            