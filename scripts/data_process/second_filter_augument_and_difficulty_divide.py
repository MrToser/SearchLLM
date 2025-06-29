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
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
# from scripts.data_process.utils import make_prefix
from search_llm.utils import make_prefix,_postprocess_responses,_example_level_pad,execute_predictions, \
    _process_next_obs,_update_prompt_state,eliminate_bad_samples
import torch

def make_map_fn(split):
    def process_fn(example, idx):
        example['question'] = example['question'].strip()
        # print("idx is:",idx)
        # print("example is:", example['id'])
        if example['question'][-1] != '?':
            example['question'] += '?'
        question = make_prefix(example, template_type=args.template_type)
        # print("question is:", question)
        # assert 1==0
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

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_1_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--second_filter', action='store_true', help='Whether to perform second filter')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        train_dataset = dataset['train']
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True,load_from_cache_file=False)
        all_dataset.append(train_dataset)
    local_dir = args.local_dir
    with open(os.path.join(local_dir, 'ids_augument.json'), 'r') as f:
        ids = json.load(f)
    # 只保留非idx部分
    all_dataset = [ds.filter(lambda x: x['id'] not in ids[x['data_source']]) for ds in all_dataset]
    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    print("len after filter is ", len(all_train_dataset))
    # assert 1==0
    # all_train_dataset = all_train_dataset.shuffle(seed=42)
    # all_train_dataset = all_train_dataset.select(range(100))  # 仅选取前1000条数据进行测试
    # assert 1==0
    # 进一步进行筛选 根据search频率
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct",
              tensor_parallel_size=1,
              gpu_memory_utilization=0.7,
              max_model_len=4096)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=512,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("len is ",len(all_train_dataset))
    
    batch_size = 512
    # new_all_train_dataset = all_train_dataset[0:512]
    # 对数据集进行分批处理
    new_ids = {}
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('jinaai/jina-embeddings-v3',trust_remote_code=True).to("cuda:1")
    
    for data_source in data_sources:
        new_ids[data_source] = {
            'search_1': [],
            'search_2': [],
            'search_3': []
        }
    for i in tqdm(range(0, len(all_train_dataset), batch_size), desc="Processing batches"):
        batch = all_train_dataset[i:i + batch_size]
        # 全局prompts
        raw_prompts = [dp[0]['content'] for dp in batch['prompt']]
        prompts_ids = batch['id']
        prompts_data_source = batch['data_source']
        # print("raw_prompts: ",raw_prompts)
        # assert 1==0
        # 使用模型生成回答
        # search过程
        batch_shape = len(raw_prompts)
        active_mask = torch.ones(batch_shape, dtype=torch.bool)
        turns_stats = torch.ones(batch_shape, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_shape, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_shape, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        print(f'-----[Debug]----- begin generation loop')
        max_turns = 5
        for step in range(max_turns):
            # print("step:", step)
            # print(f"\nstep {step}: \nraw_prompts",raw_prompts)
            if not active_mask.sum():
                break
            # actibe_mask 纬度始终为batch_size
            # 当前活跃的 prompt
            prompts_index_active = [j for j in range(batch_shape) if active_mask[j]]
            prompts = [raw_prompts[j] for j in prompts_index_active]
            # prompts_ids = [prompts_ids[j] for j in range(batch_size) if active_mask[j]]
            gen_output = llm.generate(
                prompts,
                sampling_params=sampling_params,
            )
            # print(gen_output[0:10])
            # assert 1==0
            all_generated_texts = [
                response.outputs[0].text.strip()
                for response in gen_output
            ]
            all_generated_output_ids = [
                response.outputs[0].token_ids
                for response in gen_output
            ]
            responses_ids, responses_str = _postprocess_responses(tokenizer,all_generated_output_ids)
            responses_ids, responses_str = _example_level_pad(responses_ids, responses_str, active_mask,pad_token_id=tokenizer.pad_token_id)
            next_obs, dones, valid_action, is_search = execute_predictions(
                responses_str, "", active_mask
            )
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # next_obs_ids = _process_next_obs(tokenizer,next_obs,512)
            
            # 更新raw_prompts
            new_prompts = _update_prompt_state(
                prompts,
                responses_str,
                next_obs
            )
            # 更新raw_prompts
            for j, idx in enumerate(prompts_index_active):
                raw_prompts[idx] = new_prompts[j]
            
        
        print(f"-----[Debug]----- end generation")
        # final LLM rollout
        meta_info = {}
        if active_mask.sum():
            prompts_index_active = [j for j in range(batch_shape) if active_mask[j]]
            prompts = [raw_prompts[j] for j in prompts_index_active]
            gen_output = llm.generate(
                prompts,
                sampling_params=sampling_params,
            )
            all_generated_texts = [
                response.outputs[0].text.strip()
                for response in gen_output
            ]
            all_generated_output_ids = [
                response.outputs[0].token_ids
                for response in gen_output
            ]
            responses_ids, responses_str = _postprocess_responses(tokenizer,all_generated_output_ids)
            responses_ids, responses_str = _example_level_pad(responses_ids, responses_str, active_mask,pad_token_id=tokenizer.pad_token_id)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = execute_predictions(
                responses_str, "", active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
        
            new_prompts = _update_prompt_state(
                prompts,
                responses_str,
                ['']*(len(prompts)),
            )
            # 更新raw_prompts
            for j, idx in enumerate(prompts_index_active):
                raw_prompts[idx] = new_prompts[j]    
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        print("turns_stats:", meta_info['turns_stats'])
        print("valid_action_stats:", meta_info['valid_action_stats'])
        print("valid_search_stats:", meta_info['valid_search_stats'])
        # responses = llm.generate(prompts, sampling_params=sampling_params)
        
        # 对rollout进行处理
        if args.second_filter:
            eliminate_idx = eliminate_bad_samples(
                raw_prompts, turns_stats, valid_action_stats, valid_search_stats, embed_model,prompts_ids,
            )
        else:
            eliminate_idx = []
        # print("wrong samples:",  )
        
        # 搜索一次的
        indices = (valid_search_stats <= 1 ).nonzero(as_tuple=False)
        # print("indices:", indices)
        # print(len(indices))
        if indices.numel() > 0:
            for idx_tensor in indices:
                idx = idx_tensor.item()
                if prompts_ids[idx] not in eliminate_idx:
                    new_ids[prompts_data_source[idx]]['search_1'].append(prompts_ids[idx])
        # 搜索2到3次的
        indices = ((valid_search_stats > 1 )&(valid_search_stats <= 3)).nonzero(as_tuple=False)
        # print(len(indices))
        if indices.numel() > 0:
            for idx_tensor in indices:
                idx = idx_tensor.item()
                if prompts_ids[idx] not in eliminate_idx:
                    new_ids[prompts_data_source[idx]]['search_2'].append(prompts_ids[idx])
        # 搜索3次以上的
        indices = (valid_search_stats > 3).nonzero(as_tuple=False)
        # print(len(indices))
        if indices.numel() > 0:
            for idx_tensor in indices:
                idx = idx_tensor.item()
                if prompts_ids[idx] not in eliminate_idx:
                    new_ids[prompts_data_source[idx]]['search_3'].append(prompts_ids[idx])
    # print("new_ids:", new_ids)
    for data_source in data_sources:
        print(f"\nData source: {data_source}")
        print(f"Search 1 count: {len(new_ids[data_source]['search_1'])}")
        print(f"Search 2 count: {len(new_ids[data_source]['search_2'])}")
        print(f"Search 3 count: {len(new_ids[data_source]['search_3'])}")
    import json
    if args.second_filter:
        file_name = 'ids_augument_filter_2_divide.json'
    else:
        file_name = 'ids_augument_divide.json'
    with open(os.path.join(local_dir, file_name), 'w') as f:
        json.dump(new_ids, f)

    
