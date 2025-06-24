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
from search_llm.utils import make_prefix,_postprocess_responses,_example_level_pad,execute_predictions
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_1_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='second_filter')
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
    with open(os.path.join(local_dir, 'ids.json'), 'r') as f:
        ids = json.load(f)
    # 只保留非idx部分
    all_dataset = [ds.filter(lambda x: x['id'] not in ids) for ds in all_dataset]
    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    print("len after filter is ", len(all_train_dataset))
    # assert 1==0
    # 进一步进行筛选 根据search频率
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct",tensor_parallel_size=1,gpu_memory_utilization=0.7)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=256
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("len is ",len(all_train_dataset))
    batch_size = 512
    # 对数据集进行分批处理
    ids = []
    for i in tqdm(range(0, len(all_train_dataset), batch_size), desc="Processing batches"):
        batch = all_train_dataset[i:i + batch_size]
        prompts = [dp[0]['content'] for dp in batch['prompt']]
        # 使用模型生成回答
        # search过程
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        print(f'-----[Debug]----- begin generation loop')
        max_turns = 5
        for step in range(max_turns):
            if not active_mask.sum():
                break
            gen_output = llm.generate(
                prompts,
                sampling_params=sampling_params,
            )
            all_generated_texts = [
                response.outputs[0].text.strip()
                for response in gen_output
            ]
            responses_str = _postprocess_responses(tokenizer,all_generated_texts)
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

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        print(f"-----[Debug]----- end generation")
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        
        responses = llm.generate(prompts, sampling_params=sampling_params)
        
        
        for j, response in enumerate(responses):
            generated_text = response.outputs[0].text.strip().lower()
            if generated_text == 'yes':
                ids.append(batch['id'][j])
    print("ids of the questions that can be answered:", ids)
    print(len(ids))
    # 保存ids
    import json
    with open(os.path.join(local_dir, 'ids.json'), 'w') as f:
        json.dump(ids, f)

    
