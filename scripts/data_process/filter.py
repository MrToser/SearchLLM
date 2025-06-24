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
from tqdm import tqdm
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

from search_llm.utils import make_prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_1_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='first_filter')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']
        
        def make_map_fn(split):
            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                # print("idx is:",idx)
                # print("example is:", example['id'])
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
    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    # 获得初筛的idx
    # 加载模型
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct",tensor_parallel_size=1,gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=256
    )
    print("len is ",len(all_train_dataset))
    batch_size = 2048
    # 对数据集进行分批处理
    ids = []
    for i in tqdm(range(0, len(all_train_dataset), batch_size), desc="Processing batches"):
        batch = all_train_dataset[i:i + batch_size]
        prompts = [dp[0]['content'] for dp in batch['prompt']]
        # 使用模型生成回答
        responses = llm.generate(prompts, sampling_params=sampling_params)
        for j, response in enumerate(responses):
            generated_text = response.outputs[0].text.strip().lower()
            if generated_text == 'yes':
                ids.append(batch['id'][j])
    print("ids of the questions that can be answered:", ids)
    print(len(ids))
    # 保存ids
    import json
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, 'ids.json'), 'w') as f:
        json.dump(ids, f)

    
