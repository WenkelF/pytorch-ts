from pytorch_lightning.callbacks import Callback
import torch
import numpy as np

from parse import parser
args = parser.parse_args()


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
            mean_list = torch.cat(edge_usage_list, dim=0).mean(0) # mean accross all batches (dim=0)
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
            torch.save(edges, str(args.path)+'edges.pt')
            
        return


class ReplaceEdgeCallback_upd(Callback):
    def __init__(
        self,
        num_edges: int,
        pre_epochs: int = 5,
        post_epochs: int = 5,
        mod_freq: int = 1,
        num_mods: int = 1,
    ):
        super().__init__()
        
        self.num_edges = num_edges
        self.pre_epochs = pre_epochs
        self.post_epochs = post_epochs
        self.mod_freq = mod_freq
        self.num_mods = num_mods

    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if epoch >= (self.pre_epochs - 1) and (epoch + 1 - self.pre_epochs) % self.mod_freq == 0 and epoch < trainer.max_epochs - self.post_epochs - 1:
            edge_usage_list = pl_module.edge_usage_list
            mean_list = torch.cat(edge_usage_list, dim=0).mean(0) # mean accross all batches (dim=0)
            _, indices = (-mean_list).topk(self.num_mods)

            edges = np.stack(torch.load(str(args.path)+'edges.pt'), axis=1)

            target_dim = pl_module.model.target_dim
            count = (indices < self.num_edges).sum()
            
            print("Discarded edge(s): "+str(edges[indices.cpu(),:]))
            print('Changed edges: %d/%d' % (count, self.num_mods))
            
            edges = np.delete(edges, indices.cpu(), axis=0)

            combined_edges = edges
            candidate_list = []
            while len(candidate_list) < self.num_mods:
                print("Sampling new edge candidate...")
                # sample new edge
                new_edge = np.random.choice(target_dim, 2)
                if new_edge[0] != new_edge[1]:
                    combined_edges, counts = np.unique(np.concatenate([combined_edges, new_edge.reshape(1,-1)], axis=0), axis=0, return_counts=True)
                    if counts.max() == 1:
                        print("New edge candidate: "+str(new_edge))
                        candidate_list.append(new_edge.reshape(1,-1))

            edges = np.concatenate([edges, np.concatenate(candidate_list, axis=0)], axis=0)
       
            edges = [edges[:,0].squeeze(), edges[:,1].squeeze()]
            torch.save(edges, str(args.path)+'edges.pt')
            
        return