import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 18
})

path = './Metrics/'
save_path = './figs/factor_testing/'
figsize = (16, 12)
if not os.path.exists(save_path):
    os.mkdir(save_path)

def plot_cloud_edge(cloud, edge, flag=True):
    x = np.arange(1, len(cloud) + 1)
    labels = ['' for i in range(1, len(x) + 1)]
    if flag:
        for idx in range(24, len(cloud), 24):
            labels[idx] = str(idx)

    plt.plot(x, cloud, 'o-', label='cloud')
    plt.plot(x, edge, 'o-', label='edge')

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

#############################################

price_data = {
    'cloud' : np.load(path + 'case4_price_testing_1_1_1/mvno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
    'edge' : np.load(path + 'case4_price_testing_1_1_1/mvno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
}
def plot_price_testing():    
    if not os.path.exists(save_path + 'price/'):
        os.mkdir(save_path + 'price/')
    # VoIP
    plt.figure(figsize=figsize)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 0], price_data['edge'][:, 0])
    plt.savefig(save_path + 'price/VoIP')

    # IP Video
    plt.figure(figsize=figsize)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 1], price_data['edge'][:, 1])
    plt.savefig(save_path + 'price/ipVideo')

    # FTP
    plt.figure(figsize=figsize)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 2], price_data['edge'][:, 2])
    plt.savefig(save_path + 'price/ftp')

def plot_delay_testing():
    _dir = ['case4_delay_testing_01_01_01/', 'case4_delay_testing_05_05_05/', 'case4_delay_testing_1_1_1/']
    data = []
    for d in _dir:
        data.append({
            'cloud' : np.load(path + d + 'mvno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
            'edge' : np.load(path + d + 'mvno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
        })
    if not os.path.exists(save_path + 'delay/'):
        os.mkdir(save_path + 'delay/')
    # VoIP
    plt.figure(figsize=figsize)
    ## 0
    plt.subplot(411)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - delay weighting factor: 0')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 0], price_data['edge'][:, 0], flag = False)
    ## 0.1
    plt.subplot(412)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - delay weighting factor: 0.1')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[0]['cloud'][:, 0], data[0]['edge'][:, 0], flag = False)
    ## 0.5
    plt.subplot(413)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - delay weighting factor: 0.5')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[1]['cloud'][:, 0], data[1]['edge'][:, 0], flag = False)
    ## 1
    plt.subplot(414)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - delay weighting factor: 1')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[2]['cloud'][:, 0], data[2]['edge'][:, 0])
    plt.savefig(save_path + 'delay/VoIP')

    # IP Video
    plt.figure(figsize=figsize)
    ## 0
    plt.subplot(411)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM - delay weighting factor: 0')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 1], price_data['edge'][:, 1], flag = False)
    ## 0.1
    plt.subplot(412)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM - delay weighting factor: 0.1')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[0]['cloud'][:, 1], data[0]['edge'][:, 1], flag = False)
    ## 0.5
    plt.subplot(413)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM - delay weighting factor: 0.5')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[1]['cloud'][:, 1], data[1]['edge'][:, 1], flag = False)
    ## 1
    plt.subplot(414)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM - delay weighting factor: 1')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[2]['cloud'][:, 1], data[2]['edge'][:, 1])
    plt.savefig(save_path + 'delay/ipVideo')

    # FTP
    plt.figure(figsize=figsize)
    ## 0
    plt.subplot(411)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - delay weighting factor: 0')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 2], price_data['edge'][:, 2], flag = False)
    ## 0.1
    plt.subplot(412)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - delay weighting factor: 0.1')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[0]['cloud'][:, 2], data[0]['edge'][:, 2], flag = False)
    ## 0.5
    plt.subplot(413)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - delay weighting factor: 0.5')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[1]['cloud'][:, 2], data[1]['edge'][:, 2], flag = False)
    ## 1
    plt.subplot(414)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - delay weighting factor: 1')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[2]['cloud'][:, 2], data[2]['edge'][:, 2])
    plt.savefig(save_path + 'delay/ftp')

def plot_T_testing():
    _dir = ['case4_bw_testing_05_05_05/', 'case4_bw_testing_1_1_1/', 'case4_bw_testing_5_1_3/']
    data = []
    for d in _dir:
        data.append({
            'cloud' : np.load(path + d + 'mvno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
            'edge' : np.load(path + d + 'mvno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
        })
    if not os.path.exists(save_path + 'throughput/'):
        os.mkdir(save_path + 'throughput/')
    # VoIP
    plt.figure(figsize=figsize)
    ## 0
    plt.subplot(411)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - throughput weighting factor: 0')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 0], price_data['edge'][:, 0], flag = False)
    ## 0.1
    plt.subplot(412)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - throughput weighting factor: 0.5')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[0]['cloud'][:, 0], data[0]['edge'][:, 0], flag = False)
    ## 0.5
    plt.subplot(413)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - throughput weighting factor: 1')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[1]['cloud'][:, 0], data[1]['edge'][:, 0], flag = False)
    ## 1
    plt.subplot(414)
    plt.title('number of VoIP tasks assign to MVNO cloud/edge VoIP VM - throughput weighting factor: 5')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[2]['cloud'][:, 0], data[2]['edge'][:, 0])
    plt.savefig(save_path + 'throughput/VoIP')

    # IP Video
    plt.figure(figsize=figsize)
    ## 0
    plt.subplot(311)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM - throughput weighting factor: 0')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 1], price_data['edge'][:, 1], flag = False)
    ## 0.1
    plt.subplot(312)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM - throughput weighting factor: 0.5')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[0]['cloud'][:, 1], data[0]['edge'][:, 1], flag = False)
    ## 0.5
    plt.subplot(313)
    plt.title('number of IP Video tasks assign to MVNO cloud/edge IP Video VM - throughput weighting factor: 1')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[1]['cloud'][:, 1], data[1]['edge'][:, 1])
    plt.savefig(save_path + 'throughput/ipVideo')

    # FTP
    plt.figure(figsize=figsize)
    ## 0
    plt.subplot(411)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - throughput weighting factor: 0')
    plt.ylabel('number of tasks')
    plot_cloud_edge(price_data['cloud'][:, 2], price_data['edge'][:, 2], flag = False)
    ## 0.1
    plt.subplot(412)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - throughput weighting factor: 0.5')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[0]['cloud'][:, 2], data[0]['edge'][:, 2], flag = False)
    ## 0.5
    plt.subplot(413)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - throughput weighting factor: 1')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[1]['cloud'][:, 2], data[1]['edge'][:, 2], flag = False)
    ## 1
    plt.subplot(414)
    plt.title('number of FTP tasks assign to MVNO cloud/edge FTP VM - throughput weighting factor: 3')
    plt.xlabel('hour')
    plt.ylabel('number of tasks')
    plot_cloud_edge(data[2]['cloud'][:, 2], data[2]['edge'][:, 2])
    plt.savefig(save_path + 'throughput/ftp')

plot_price_testing()
plot_delay_testing()
plot_T_testing()