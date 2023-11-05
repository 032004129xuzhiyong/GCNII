# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月29日
"""
import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mytool import tool
from mytool import mytorch as mtorch
from mytool import metric as mmetric
from mytool import plot as mplot
from datasets.dataset import load_mat
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def bind_boolind_for_fn(func, train_bool_ind, val_bool_ind):
    def binded_func(scores, labels):
        if scores.requires_grad == True:
            return func(scores[train_bool_ind], labels[train_bool_ind])
        else:
            return func(scores[val_bool_ind], labels[val_bool_ind])

    tool.set_func_name(binded_func, tool.get_func_name(func))
    return binded_func


def train_one_args(args):
    # load data
    adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class = load_mat(**args['dataset_args'])
    dataload = [((inputs, adjs), labels)]

    # build model
    # origin model
    device = args['device']
    ModelClass = tool.import_class(**args['model_class_args'])
    model = ModelClass(n_feats=sum(n_feats), n_class=n_class, **args['model_args']).to(device)

    # loss optimizer lr_scheduler
    loss_fn = nn.CrossEntropyLoss()
    OptimizerClass = tool.import_class(**args['optimizer_class_args'])
    optimizer = OptimizerClass([
        dict(params=model.reg_params, weight_decay=args['weight_decay1']),
        dict(params=model.non_reg_params, weight_decay=args['weight_decay2'])
    ], **args['optimizer_args'])
    # SchedulerClass = tool.import_class(**args['scheduler_class_args'])
    # scheduler = SchedulerClass(optimizer,**args['scheduler_args'])
    # warpscheduler
    # def sche_func(epoch,lr,epoch_logs):
    #    scheduler.step(epoch_logs['val_loss'])
    # scheduler_callback = mtorch.SchedulerWrapCallback(sche_func,True)


    # training
    history = mtorch.fit(model, dataload, epochs=args['epochs'],
                         compile_kw={'loss': bind_boolind_for_fn(loss_fn, train_bool, val_bool),
                                     'optimizer': optimizer,
                                     'metric': bind_boolind_for_fn(mmetric.acc, train_bool, val_bool),
                                     # 'scheduler':scheduler,
                                     },
                         device=device,
                         val_dataload=dataload,
                         loss_weights=None,
                         callbacks=[
                             mtorch.DfSaveCallback(**args['dfcallback_args']),
                             #mtorch.TbWriterCallback(**args['tbwriter_args']),
                             mtorch.EarlyStoppingCallback(**args['earlystop_args']),
                             mtorch.TunerRemovePreFileInDir([
                                 args['earlystop_args']['checkpoint_dir'],
                                 #args['tbwriter_args']['log_dir'],
                             ], 10, 0.8)
                             # mtorch.PlotLossMetricTimeLr(),
                             # scheduler_callback,
                         ])

    # return
    return history.history


def train_with_besthp_and_save_config_and_history(best_conf):
    """
    保存两个数据： 最优配置(存储为yaml文件) 和  多次实验的过程数据(pd.DataFrame数据格式存储为多个csv文件)
    :param best_conf: dict
    :return:
        None
    """
    best_dir = best_conf['best_trial_save_dir']
    dataset_name = tool.get_basename_split_ext(best_conf['dfcallback_args']['df_save_path'])
    best_dataset_dir = os.path.join(best_dir, dataset_name)
    if not os.path.exists(best_dataset_dir):
        os.makedirs(best_dataset_dir)

    for tri_idx in range(best_conf['best_trial']):
        tri_logs = train_one_args(best_conf)
        df = pd.DataFrame(tri_logs)
        #csv
        df_save_path = os.path.join(best_dataset_dir, 'df' + str(tri_idx) + '.csv')
        df.to_csv(df_save_path, index=False, header=True)
    mean_acc, std_acc = compute_mean_val_acc_in_bestdir_for_one_dataset(best_dataset_dir,True)

    best_conf.update({
        'acc_mean': float(mean_acc),
        'acc_std': float(std_acc),
    })
    save_conf_path = os.path.join(best_dataset_dir, 'conf.yaml')
    tool.save_yaml_args(save_conf_path, best_conf)

def compute_mean_val_acc_in_bestdir_for_one_dataset(one_dataset_dir, if_plot_fig=False):
    """
    计算一个数据集多次实验的平均acc 和 std
    :param one_dataset_dir: 包含csv过程数据的目录
    :return:
        (mean, std)
    """
    filenames = os.listdir(one_dataset_dir)
    filenames = [name for name in filenames if name.endswith('csv')]
    filepaths = [os.path.join(one_dataset_dir, name) for name in filenames]

    val_metric_list = []
    for fp in filepaths:
        df = pd.read_csv(fp)
        # png
        if if_plot_fig:
            fig = mplot.plot_LossMetricTimeLr_with_df(df)
            fig.savefig(os.path.join(one_dataset_dir, tool.get_basename_split_ext(fp) + '.png'))
            plt.close()  # 关闭figure
            del fig
        val_acc = df['val_metric_acc'].to_numpy().max()
        del df
        val_metric_list.append(val_acc)
    val_metric_list = np.array(val_metric_list)
    return np.mean(val_metric_list), np.std(val_metric_list)


def compute_mean_val_acc_in_bestdir_for_all_dataset(best_dir):
    """
    计算best目录下所有数据集 mean_acc 和 std_acc
    :param best_dir:
    :return:
        Dict[datasetname, (mean_acc, std_acc)]
    """
    dataset_names = os.listdir(best_dir)
    dataset_dirs = [os.path.join(best_dir, dn) for dn in dataset_names]
    dataset_mean_std = [compute_mean_val_acc_in_bestdir_for_one_dataset(ddir) for ddir in dataset_dirs]
    return dict(zip(dataset_names, dataset_mean_std))


class MyTuner(mtorch.MyTuner):
    """hyperparameters search"""

    def run_trial(self, hp, **kwargs):
        args = kwargs['args']  # dict
        args = tool.modify_dict_with_hp(args, hp)
        history = train_one_args(args)
        return max(history['val_metric_acc'])


if __name__ == '__main__':
    torch.cuda.empty_cache()
    config_list = [
        # 'config/ALOI.yaml', #完成
        # 'config/3sources.yaml', #完成
        # 'config/BBCnews.yaml', #完成
        # 'config/BBCSports.yaml', #完成
        # 'config/citeseer.yaml', #完成
        # 'config/Cora.yaml',
        # 'config/MNIST.yaml', #完成
        # 'config/NGs.yaml',
         'config/Wikipedia.yaml',
    ]
    #打印
    for conf in config_list:
        print(conf,end=' ')
    print('will train!')

    for conf in config_list:
        args = tool.load_yaml_args(conf)
        args['dfcallback_args']['df_save_path'] = os.path.join('./tables/', tool.get_basename_split_ext(conf) + '.csv')
        args['tbwriter_args']['log_dir'] = os.path.join('./logs/',tool.get_basename_split_ext(conf))
        args['earlystop_args']['checkpoint_dir'] = os.path.join('./checkpoint/',tool.get_basename_split_ext(conf))
        origin_args = copy.deepcopy(args)

        # tuner
        tuner = MyTuner(
            executions_per_trial=args['executions_per_trial'],
            max_trials=args['max_trials'],
            mode='max',
        )
        tuner.search(args=args)
        torch.cuda.empty_cache()

        # 获得最优config
        best_hp = tuner.get_best_hyperparameters()[0]
        #打印
        for i in range(5):
            print('*' * 50)
        tool.print_dicts_tablefmt([best_hp],['Best HyperParameters!!'])
        for i in range(5):
            print('*' * 50)
        best_args = tool.modify_dict_with_hp(origin_args, best_hp, False)

        # 用最优参数训练，评估平均准确率，并保存实验过程数据和最优配置在 ./best目录下
        train_with_besthp_and_save_config_and_history(best_args)
        torch.cuda.empty_cache()
    print(compute_mean_val_acc_in_bestdir_for_all_dataset('best'))
    
    #print(compute_mean_val_acc_in_bestdir_for_one_dataset('best/MNIST',True))

