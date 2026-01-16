import torch
print(torch.cuda.is_available())  # gpu testing code
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
