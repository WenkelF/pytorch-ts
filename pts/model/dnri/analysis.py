import torch

edge_usage_list = torch.load('./checkpoints/edge_usage.pt')

# print(len(edge_usage_list))
# print(edge_usage_list[0].shape)
# print(edge_usage_list[0][:,:,1].squeeze().shape)

# edge_usage_list = edge_usage_list[0][:,:,1].squeeze()

# per time step
print("Per time step:")
mean_list = torch.stack(edge_usage_list, dim=2).mean(0)
std_list = torch.stack(edge_usage_list, dim=2).std(0)
# print(mean_list)
# print(std_list)
print("mean:")
print(mean_list.min().item())
print(mean_list.mean().item())
print(mean_list.max().item())
print("std:")
print(std_list.min().item())
print(std_list.mean().item())
print(std_list.max().item())

# per time series
print("Per time series:")
mean_list = torch.stack(edge_usage_list, dim=2).mean(2)
std_list = torch.stack(edge_usage_list, dim=2).std(2)
# print(mean_list)
# print(std_list)
print("mean:")
print(mean_list.min().item())
print(mean_list.mean().item())
print(mean_list.max().item())
print("std:")
print(std_list.min().item())
print(std_list.mean().item())
print(std_list.max().item())

# overall
print("Overall:")
mean_list = torch.cat(edge_usage_list, dim=0).mean(0)
std_list = torch.cat(edge_usage_list, dim=0).std(0)
# print(mean_list)
# print(std_list)
print("mean:")
print(mean_list.min().item())
print(mean_list.mean().item())
print(mean_list.max().item())
print("std:")
print(std_list.min().item())
print(std_list.mean().item())
print(std_list.max().item())