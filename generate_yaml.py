# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年11月13日
"""

from mytool import tool
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-paths', '-dps',
                        nargs='+',
                        type=str,
                        required=True,
                        help='which dataset config file will be generated',
                        dest='dps')
    parser.add_argument('--template-config-path', '-tcp',
                        default='config/3sources.yaml',
                        type=str,
                        help='generate config yaml from it which is a yaml file path',
                        dest='tcp')
    parser_args = vars(parser.parse_args())
    for dataset_path in parser_args['dps']:
        if os.path.exists(dataset_path):
            template_config = tool.load_yaml_args(parser_args['tcp'])
            template_config['dataset_args']['mat_path'] = dataset_path
            tool.save_yaml_args(os.path.join('config/',
                                tool.get_basename_split_ext(dataset_path)+'.yaml'),
                                template_config)
        else:
            raise Exception('dataset path does not exist!!')

