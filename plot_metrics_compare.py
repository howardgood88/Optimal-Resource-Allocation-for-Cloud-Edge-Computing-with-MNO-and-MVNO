from parameters import case_num, expected_task_num
import numpy as np
import matplotlib.pyplot as plt
import os

data = []
path = f'./Metrics/{case_num}'
data.append({
    # 'mno_vm_resource' : np.load(path + 'mno_vm_resource.npy'), # 3x3
    # 'mvno_vm_resource' : np.load(path + 'mvno_vm_resource.npy'), # 3x3
    # 'mvno_vm_cost' : np.load(path + 'mvno_vm_cost.npy'), # float
    # 'mno_task_fitness' : np.load(path + 'mno_task_fitness.npy'), # (VoIP, IP Video, FTP)
    # 'mno_task_resource' : np.load(path + 'mno_task_resource.npy'), # 3x3
    # 'mvno_task_fitness' : np.load(path + 'mvno_task_fitness.npy'), # (VoIP, IP Video, FTP)
    # 'mvno_task_resource' : np.load(path + 'mvno_task_resource.npy'), # 3x3
    # 'mno_block_rate' : np.load(path + 'mno_block_rate.npy'), # (VoIP, IP Video, FTP)
    # 'mvno_block_rate' : np.load(path + 'mvno_block_rate.npy'), # (VoIP, IP Video, FTP)
    # 'mno_user_cost' : np.load(path + 'mno_user_cost.npy'), # float
    # 'mvno_user_cost' : np.load(path + 'mvno_user_cost.npy'), # float
    # 'mno_cloud_task_num' : np.load(path + 'mno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
    # 'mno_edge_task_num' : np.load(path + 'mno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
    # 'mvno_cloud_task_num' : np.load(path + 'mvno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
    # 'mvno_edge_task_num' : np.load(path + 'mvno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
    'mno_profit' : np.load(path + 'mno_profit.npy'), # float
    'mvno_profit' : np.load(path + 'mvno_profit.npy'), # float
})
path = f'./no_mvno/Metrics/{case_num}'
data.append({
    # 'mno_task_fitness' : np.load(path + 'mno_task_fitness.npy'), # (VoIP, IP Video, FTP)
    # 'mno_task_resource' : np.load(path + 'mno_task_resource.npy'), # 3x3
    # 'mno_block_rate' : np.load(path + 'mno_block_rate.npy'), # (VoIP, IP Video, FTP)
    # 'mno_user_cost' : np.load(path + 'mno_user_cost.npy'), # float
    # 'mno_cloud_task_num' : np.load(path + 'mno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
    # 'mno_edge_task_num' : np.load(path + 'mno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
    'mno_profit' : np.load(path + 'mno_profit.npy'), # float
})

_dir = f'./figs/comparison/'
if not os.path.exists(_dir):
    os.makedirs(_dir)

def plot_profit():
    plt.figure(figsize=(16,12))
    plt.title('Profit comparison between Only MNO, MNO with MVNO and MVNO with MNO in each round')
    plt.xlabel('round')
    plt.ylabel('profit (dollar)')
    x = np.arange(1, len(data[1]['mno_profit']) + 1)
    plt.bar(x - 0.2, data[1]['mno_profit'], width = 0.2, label='Only MNO')
    plt.bar(x, data[0]['mno_profit'], width = 0.2, label='MNO with MVNO')
    plt.bar(x + 0.2, data[0]['mvno_profit'], width = 0.2, label='MVNO with MNO')
    plt.legend()
    plt.savefig(_dir + f'profit_{expected_task_num}_03_09')
    plt.show()

########################## 
plot_profit()

print(f'Save figs to {_dir}!')