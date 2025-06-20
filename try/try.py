# import matplotlib.pyplot as plt
# import math

# # Parameters
# total_steps = 300
# min_lr_ratio = 0.05

# # Define the cosine decay function
# def cosine_decay_lambda(total_steps, min_lr_ratio=0.05):
#     def lr_lambda(current_step):
#         progress = min(current_step / total_steps, 1.0)
#         decay = 1 - (2*progress ** 2)
#         return max(min_lr_ratio, decay)
#     return lr_lambda

# # Generate values
# lr_func = cosine_decay_lambda(total_steps, min_lr_ratio)
# steps = list(range(total_steps + 1))
# lr_values = [lr_func(step) for step in steps]

# # Plotting

# plot_path = "./1"
# plt.figure(figsize=(8, 4))
# plt.plot(steps, lr_values, label='Cosine Decay with Min LR Ratio')
# plt.xlabel('Training Step')
# plt.ylabel('Learning Rate Scale')
# plt.title('Cosine Decay Learning Rate Schedule')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(plot_path)
# main_ppo.py
# 临时跳过设置 CUDA 种子
import torch
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(20)
    print()

import torch; 
print(torch.version.hip); 
print(torch.cuda.device_count())
print(torch.cuda.is_available())