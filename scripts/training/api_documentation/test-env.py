import torch
print(torch.cuda.is_available())         # Should be True
print(torch.version.cuda)                # Should match 12.1
print(torch.cuda.get_device_name(0))     # Should show your GPU
