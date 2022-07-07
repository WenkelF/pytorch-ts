from pytorch_lightning.callbacks import Callback
import torch
import numpy as np


class EdgeUsageCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        torch.save(pl_module.edge_usage_list, './checkpoints/edge_usage.pt')
        
        return
    
class EpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("Epoch : "+str(pl_module.current_epoch))
        
        return



class ReplaceEdgeCallback(Callback):
    def __init__(
        self,
        pre_epochs: int = 5,
        post_epochs: int = 5,
        mod_freq: int = 1,
        num_mods: int = 1,
    ):
        super().__init__()
        
        self.pre_epochs = pre_epochs
        self.post_epochs = post_epochs
        self.mod_freq = mod_freq
        self.num_mods = num_mods
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if epoch >= (self.pre_epochs - 1) and (epoch + 1 - self.pre_epochs) % self.mod_freq == 0 and epoch < trainer.max_epochs - self.post_epochs - 1:
            edge_usage_list = pl_module.edge_usage_list
            mean_list = torch.cat(edge_usage_list, dim=0).mean(0)
            _, edge_idx_list = (-mean_list).topk(self.num_mods)
            
            target_dim = pl_module.model.target_dim
            edges = np.stack(pl_module.model.edges, axis=1)
            num_edges = len(edges)
            
            print("Old edge(s): "+str(edges[edge_idx_list.cpu(),:]))
            
            edges = np.delete(edges, edge_idx_list.cpu(), axis=0)
            
            while len(edges) < num_edges:
                print("Sampling new edge...")
                # sample new edge
                new_edge = np.random.choice(target_dim, 2)
                if new_edge[0] != new_edge[1]:
                    edges, counts = np.unique(np.concatenate([edges, new_edge.reshape(1,-1)], axis=0), axis=0, return_counts=True)
                    if counts.max() == 1:
                        print("New edge: "+str(new_edge))
                        
            edges = [edges[:,0].squeeze(), edges[:,1].squeeze()]
            torch.save(edges, './checkpoints/edges.pt')
            
        return




class ReplaceEdgeCallback_v1(Callback):
    def __init__(
        self,
        pre_epochs: int = 5,
        post_epochs: int = 5,
        mod_freq: int = 1,
    ):
        super().__init__()
        
        self.pre_epochs = pre_epochs
        self.post_epochs = post_epochs
        self.mod_freq = mod_freq
        self.num_mods = num_mods
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if epoch >= (self.pre_epochs - 1) and (epoch + 1 - self.pre_epochs) % self.mod_freq == 0 and epoch < trainer.max_epochs - self.post_epochs - 1:
            edge_usage_list = pl_module.edge_usage_list
            mean_list = torch.cat(edge_usage_list, dim=0).mean(0)
            edge_idx = mean_list.argmin()
            
            target_dim = pl_module.model.target_dim
            edges = np.stack(pl_module.model.edges, axis=1)
            num_edges = len(edges)
            
            print("Old edge: "+str(edges[edge_idx,:]))
            
            edges = np.delete(edges, edge_idx.cpu(), axis=0)
            
            while len(edges) < num_edges:
                print("Sampling new edge...")
                # sample new edge
                new_edge = np.random.choice(target_dim, 2)
                if new_edge[0] != new_edge[1]:
                    edges = np.unique(np.concatenate([edges, new_edge.reshape(1,-1)], axis=0), axis=0)

            print("New edge: "+str(new_edge))
            edges = [edges[:,0].squeeze(), edges[:,1].squeeze()]
            torch.save(edges, './checkpoints/edges.pt')
            # print("New edges: "+str(edges))
            
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
class ReplaceEdgeCallback_v0(Callback):
    def on_train_start(self, trainer, pl_module):
        torch.save(pl_module.model.edges, './checkpoints/edges.pt')
        
        return    
    
    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch > 0:   
            pl_module.model.edges = torch.load('./checkpoints/edges.pt')
            print("Used edges: "+str(pl_module.model.edges))
            
        return
    
    def on_train_epoch_end(self, trainer, pl_module):
        edge_usage_list = pl_module.edge_usage_list
        mean_list = torch.cat(edge_usage_list, dim=0).mean(0)
        edge_idx = mean_list.argmin()
        
        target_dim = pl_module.model.target_dim
        edges = np.stack(pl_module.model.edges, axis=1)
        num_edges = len(edges)
        
        print("Old edge: "+str(edges[edge_idx,:]))
        
        edges = np.delete(edges, edge_idx.cpu(), axis=0)
        
        while len(edges) < num_edges:
            print("Sampling new edge...")
            # sample new edge
            new_edge = np.random.choice(target_dim, 2)
            if new_edge[0] != new_edge[1]:
                edges = np.unique(np.concatenate([edges, new_edge.reshape(1,-1)], axis=0), axis=0)

        print("New edge: "+str(new_edge))
        edges = [edges[:,0].squeeze(), edges[:,1].squeeze()]
        torch.save(edges, './checkpoints/edges.pt')
        print("New edges: "+str(edges))
            
        return