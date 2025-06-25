# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from search_llm.utils import make_prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--difficulty', type=str, default='medium')
    # parser.add_argument('--filter', action='store_true', help='Whether to filter the dataset based on ids')
    parser.add_argument('--ids_file', type=str, default=None, help='Path to the filter ids file')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []
    difficulties = args.difficulty.split(',')
    
    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # 只加载有效数据
    import json
    # if args.filter:
    #     file_name = 'ids_augument_filter.json'
    # else:
    #     file_name = 'ids_augument.json'
    with open(os.path.join(local_dir, args.ids_file), 'r') as f:
        ids = json.load(f)
    
    # 筛选有效数据集
    difficulty_name = ""
    for difficulty in difficulties:
        if difficulty == 'easy':
            all_dataset = [ds.filter(lambda x: x['id'] in ids[x['data_source']]['search_1']) for ds in all_dataset]
        elif difficulty == 'medium':
            all_dataset = [ds.filter(lambda x: x['id'] in ids[x['data_source']]['search_2']) for ds in all_dataset]
        elif difficulty == 'hard':
            all_dataset = [ds.filter(lambda x: x['id'] in ids[x['data_source']]['search_3']) for ds in all_dataset]
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")
        difficulty_name+= f"_{difficulty}"
    # if args.difficulty == 'medium':
    #     all_dataset = [ds.filter(lambda x: x['id'] in ids[x['data_source']]['search_2']) for ds in all_dataset]
    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train'+difficulty_name+'.parquet'))

    print(f"Total number of training samples: {len(all_train_dataset)}")
    # assert 1==0
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
