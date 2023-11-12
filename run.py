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
import argparse

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
    if device=='tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
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
                                     'metric': [bind_boolind_for_fn(mmetric.acc, train_bool, val_bool),
                                                bind_boolind_for_fn(mmetric.f1, train_bool, val_bool),
                                                bind_boolind_for_fn(mmetric.precision,train_bool,val_bool),
                                                bind_boolind_for_fn(mmetric.recall,train_bool,val_bool)
                                                ],
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
    #{key1: [mean, std], key2: [mean, std]...}
    mean_std_metric_dict = compute_mean_metric_in_bestdir_for_one_dataset(best_dataset_dir, if_plot_fig=True)
    #{key1:{mean:float,std:float}, key2:{mean:float,std:float}...}
    mean_std_metric_dict = {key: {'mean':mean_std_metric_dict[key][0],'std':mean_std_metric_dict[key][1]}
                            for key in mean_std_metric_dict.keys()}
    best_conf.update(mean_std_metric_dict)
    save_conf_path = os.path.join(best_dataset_dir, 'conf.yaml')
    tool.save_yaml_args(save_conf_path, best_conf)


def compute_mean_metric_in_bestdir_for_one_dataset(one_dataset_dir, if_plot_fig=False):
    """

    Args:
        one_dataset_dir: str
        if_plot_fig: a plot every df

    Returns:
        mean_std_metric_dict: Dict {key1:[mean,std],key2:[mean,std]...}
    """
    filenames = os.listdir(one_dataset_dir)
    filenames = [name for name in filenames if name.endswith('csv')]
    filepaths = [os.path.join(one_dataset_dir, name) for name in filenames]

    metric_list = mtorch.History() #{key1:[], key2:[]...}
    for fp in filepaths:
        df = pd.read_csv(fp)
        df_col_names = df.columns
        # png
        if if_plot_fig:
            fig = mplot.plot_LossMetricTimeLr_with_df(df)
            fig.savefig(os.path.join(one_dataset_dir, tool.get_basename_split_ext(fp) + '.png'))
            plt.close()  # 关闭figure
            del fig
        metric_dict = df.iloc[:,df_col_names.str.contains('metric')].max(axis=0).to_dict()
        metric_list.update(metric_dict)
    metric_list = metric_list.history
    #{key1:[mean,std],key2:[mean,std]...}
    mean_std_metric_dict = {key:[float(np.mean(metric_list[key])), float(np.std(metric_list[key]))]
                            for key in metric_list.keys()}
    return mean_std_metric_dict


def compute_mean_metric_in_bestdir_for_all_dataset(best_dir):
    """
    计算best目录下所有数据集 mean_acc 和 std_acc
    :param best_dir:
    :return:
        Dict[datasetname, (mean_acc, std_acc)]
    """
    dataset_names = os.listdir(best_dir)
    dataset_dirs = [os.path.join(best_dir, dn) for dn in dataset_names]
    dataset_mean_std = [compute_mean_metric_in_bestdir_for_one_dataset(ddir) for ddir in dataset_dirs]
    return dict(zip(dataset_names, dataset_mean_std))


class MyTuner(mtorch.MyTuner):
    """hyperparameters search"""

    def run_trial(self, hp, **kwargs):
        args = kwargs['args']  # dict
        args = tool.modify_dict_with_hp(args, hp)
        history = train_one_args(args)
        return max(history['val_metric_acc'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        nargs='+',
                        required=True,
                        help='yaml config file path. e.g. config/3sources.yaml')
    #tuner
    parser.add_argument('--max-trials',
                        default=250,
                        type=int,
                        help='最大实验次数',
                        dest='max_trials')
    parser.add_argument('--executions-per-trial',
                        default=5,
                        type=int,
                        help='每次实验(每个配置)执行几遍，减少误差',
                        dest='executions_per_trial')
    parser.add_argument('--best-trial',
                        default=20,
                        type=int,
                        help='获得最优超参数后，使用超参数执行几次来获得实验数据',
                        dest='best_trial')
    parser.add_argument('--best-trial-save-dir',
                        default='best/',
                        type=str,
                        help='最优超参数实验数据存储目录',
                        dest='best_trial_save_dir')
    #dataset
    parser.add_argument('--topk',
                        default=10,
                        type=int,
                        help='knn topk')
    parser.add_argument('--train-ratio',
                        default=0.05,
                        type=float,
                        help='train val split',
                        dest='train_ratio')
    #model args
    parser.add_argument('--layerclass',
                        default='GCNIILayer',
                        type=str,
                        help='GCNII layer class, all layer classes: GCNIILayer/GCNII_star_Layer')
    parser.add_argument('--nlayer',
                        default=None,
                        type=int,
                        help='num layerclass')
    parser.add_argument('--hid-dim',
                        default=None,
                        type=int,
                        help='hidden dimension',
                        dest='hid_dim')
    #training
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help='torch device')
    parser.add_argument('--epochs',
                        default=1500,
                        type=int,
                        help='epochs per training')
    parser.add_argument('--save-best-only', '-sbo',
                        action='store_true',
                        default=False,
                        help='earlystop args if save best only',
                        dest='save_best_only')
    parser.add_argument('--monitor',
                        default='val_metric_acc',
                        type=str,
                        help='earlystop args monitor metrics. e.g. loss/val_loss/metric_acc/val_metric_acc')
    parser.add_argument('--patience',
                        default=100,
                        type=int,
                        help='earlystop args patience')
    #others
    parser.add_argument('--train-times-with-no-tuner',
                        default=1,
                        type=int,
                        help='训练实验次数，没有超参数搜索(默认有超参数搜索)',
                        dest='tt_nt')
    parser.add_argument('--train-save-dir-with-no-tuner',
                        default='temp_result/',
                        type=str,
                        help='训练实验数据保存目录，没有超参数搜索(默认有超参数搜索)',
                        dest='tsd_nt')

    parser_args = vars(parser.parse_args())
    # print(parser_args)
    for conf in parser_args['config']:
        print(conf, end=' ')
    print('will train!')

    for conf in parser_args['config']:
        args = tool.load_yaml_args(conf)
        #修改保存路径
        args['dfcallback_args']['df_save_path'] = os.path.join('./tables/', tool.get_basename_split_ext(conf) + '.csv')
        args['tbwriter_args']['log_dir'] = os.path.join('./logs/', tool.get_basename_split_ext(conf))
        args['earlystop_args']['checkpoint_dir'] = os.path.join('./checkpoint/', tool.get_basename_split_ext(conf))

        #根据parser_args修改conf
        #config yaml 第一层级
        first_deep_dict = {key:parser_args[key] for key in parser_args.keys()
                           if key in ['max_trials','executions_per_trial',
                            'best_trial','best_trial_save_dir','device','epochs']}
        args.update(first_deep_dict)
        #config yaml 第二层级
        args['dataset_args']['topk'] = parser_args['topk']
        args['dataset_args']['train_ratio'] = parser_args['train_ratio']
        args['model_args']['layerclass'] = parser_args['layerclass']
        args['model_args']['nlayer'] = parser_args['nlayer']  if parser_args['nlayer'] is not None else args['model_args']['nlayer']
        args['model_args']['hid_dim'] = parser_args['hid_dim'] if parser_args['hid_dim'] is not None else args['model_args']['hid_dim']
        args['earlystop_args']['save_best_only'] = parser_args['save_best_only']
        args['earlystop_args']['monitor'] = parser_args['monitor']
        args['earlystop_args']['patience'] = parser_args['patience']

        if tool.has_hyperparameter(args):
            #tuner
            # for train_with_besthp_and_save_config_and_history
            origin_args = copy.deepcopy(args)

            # tuner
            tuner = MyTuner(
                executions_per_trial=args['executions_per_trial'],
                max_trials=args['max_trials'],
                mode='max',
            )
            tuner.search(args=args)

            # 获得最优config
            best_hp = tuner.get_best_hyperparameters()[0]
            # 打印
            for i in range(5):
                print('*' * 50)
            tool.print_dicts_tablefmt([best_hp], ['Best HyperParameters!!'])
            for i in range(5):
                print('*' * 50)

            #用最优参数训练，评估平均准确率，并保存实验过程数据和最优配置在 ./best目录下
            best_args = tool.modify_dict_with_hp(origin_args, best_hp, False)
            train_with_besthp_and_save_config_and_history(best_args)
        else:
            #only train times
            #没有超参数搜索
            best_args = args
            #修改用于最优参数训练的参数
            best_args['best_trial'] = parser_args['tt_nt']
            best_args['best_trial_save_dir'] = parser_args['tsd_nt']
            train_with_besthp_and_save_config_and_history(best_args)


    print(compute_mean_metric_in_bestdir_for_all_dataset('temp_result'))

    #print(compute_mean_metric_in_bestdir_for_one_dataset('best/NGs',True))

