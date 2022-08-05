from parameters import case_num, expected_task_num
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 24,
})

data = []
path = f'./Metrics/{case_num}{expected_task_num}/'
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
    'mno_revenue' : np.load(path + 'mno_revenue.npy'), # float
    'mno_cost' : np.load(path + 'mno_cost.npy'), # float
    'mvno_revenue' : np.load(path + 'mvno_revenue.npy'), # float
    'mvno_cost' : np.load(path + 'mvno_cost.npy'), # float
})
path = f'./no_mvno/Metrics/{case_num}{expected_task_num}/'
data.append({
    # 'mno_task_fitness' : np.load(path + 'mno_task_fitness.npy'), # (VoIP, IP Video, FTP)
    # 'mno_task_resource' : np.load(path + 'mno_task_resource.npy'), # 3x3
    # 'mno_block_rate' : np.load(path + 'mno_block_rate.npy'), # (VoIP, IP Video, FTP)
    # 'mno_user_cost' : np.load(path + 'mno_user_cost.npy'), # float
    # 'mno_cloud_task_num' : np.load(path + 'mno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
    # 'mno_edge_task_num' : np.load(path + 'mno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
    'mno_revenue' : np.load(path + 'mno_revenue.npy'), # float
    'mno_cost' : np.load(path + 'mno_cost.npy'), # float
})
# print(np.average(data[1]['mno_revenue']))
# print(np.average(data[1]['mno_cost']))
# print(np.average(data[1]['mno_revenue'] - data[1]['mno_cost']))
# print(np.sum(data[1]['mno_revenue'] - data[1]['mno_cost']) / np.sum(data[1]['mno_revenue']))
# assert()

max_y = max(max(data[1]['mno_revenue']), max(data[0]['mno_revenue']), max(data[0]['mvno_revenue']),
            max(data[1]['mno_cost']), max(data[0]['mno_cost']), max(data[0]['mvno_cost']),
            max(data[1]['mno_revenue'] - data[1]['mno_cost']), max(data[0]['mno_revenue'] - data[0]['mno_cost']), max(data[0]['mvno_revenue'] - data[0]['mvno_cost']))
min_y = min(min(data[1]['mno_revenue']), min(data[0]['mno_revenue']), min(data[0]['mvno_revenue']),
            min(data[1]['mno_cost']), min(data[0]['mno_cost']), min(data[0]['mvno_cost']),
            min(data[1]['mno_revenue'] - data[1]['mno_cost']), min(data[0]['mno_revenue'] - data[0]['mno_cost']), min(data[0]['mvno_revenue'] - data[0]['mvno_cost']))

_dir = f'./figs/comparison/{expected_task_num}/'
if not os.path.exists(_dir):
    os.makedirs(_dir)

def plot_revenue():
    plt.figure(figsize=(16,12))
    plt.title('Revenue comparison in each round')
    plt.xlabel('round (day)')
    plt.ylabel('total revenue (NTD)')
    x = np.arange(1, len(data[1]['mno_revenue']) + 1)
    plt.bar(x - 0.2, data[1]['mno_revenue'], width = 0.2, label='Only MNO')
    plt.bar(x, data[0]['mno_revenue'], width = 0.2, label='MNO')
    plt.bar(x + 0.2, data[0]['mvno_revenue'], width = 0.2, label='MVNO')
    plt.ylim(min_y, max_y)
    plt.legend()
    plt.savefig(_dir + 'revenue')
    # plt.show()

def plot_cost():
    plt.figure(figsize=(16,12))
    plt.title('Cost comparison in each round')
    plt.xlabel('round (day)')
    plt.ylabel('total cost (NTD)')
    x = np.arange(1, len(data[1]['mno_cost']) + 1)
    plt.bar(x - 0.2, data[1]['mno_cost'], width = 0.2, label='Only MNO')
    plt.bar(x, data[0]['mno_cost'], width = 0.2, label='MNO')
    plt.bar(x + 0.2, data[0]['mvno_cost'], width = 0.2, label='MVNO')
    plt.ylim(min_y, max_y)
    plt.legend()
    plt.savefig(_dir + 'cost')
    # plt.show()

def plot_profit():
    plt.figure(figsize=(16,12))
    plt.title('Profit comparison in each round')
    plt.xlabel('round (day)')
    plt.ylabel('total profit (NTD)')
    x = np.arange(1, len(data[1]['mno_revenue']) + 1)
    plt.bar(x - 0.2, data[1]['mno_revenue'] - data[1]['mno_cost'], width = 0.2, label='Only MNO')
    plt.bar(x, data[0]['mno_revenue'] - data[0]['mno_cost'], width = 0.2, label='MNO')
    plt.bar(x + 0.2, data[0]['mvno_revenue'] - data[0]['mvno_cost'], width = 0.2, label='MVNO')
    plt.ylim(min_y, max_y)
    plt.legend()
    plt.savefig(_dir + 'profit')
    # plt.show()

########################## 
plot_revenue()
plot_cost()
plot_profit()

print(f'Save figs to {_dir}!')