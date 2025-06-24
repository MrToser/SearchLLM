import json
import datasets
from tqdm import tqdm
import torch
from typing import List, Tuple, Any
import re  
import requests 

def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'first_filter':
        """This works for any base model"""
        prefix = f"Could you tell me if you have enough knowledge to answer the following question? \
Just tell me yes or no, don't answer the question. \
\nQuestion: {question}\nAnswer: "
    elif template_type == 'second_filter':
        """This works for Qwen2.5-3B-Instruct"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix

def _example_level_pad(responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor,
                          pad_token_id) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        assert active_mask.sum() == responses.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses
        
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str

def _postprocess_responses(tokenizer, responses: torch.Tensor) -> torch.Tensor:
    """Process responses to stop at search operation or answer operation."""
    # responses_str = tokenizer.batch_decode(
    #     responses, 
    #     skip_special_tokens=True
    # )

    responses_str = [resp.split('</search>')[0] + '</search>'
             if '</search>' in resp 
             else resp.split('</answer>')[0] + '</answer>'
             if '</answer>' in resp 
             else resp
             for resp in responses_str]
    
    return responses_str


def postprocess_predictions(predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = [] 
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

def _batch_search(queries):
    # print("-----[Debug]----- batch_search queries:", queries)
    payload = {
        "queries": queries,
        "topk": 3,
        "return_scores": True
    }
    for _ in range(100):
        try:
            my_respones = requests.post("http://127.0.0.1:8002/retrieve", json=payload).json()
            break
        except Exception as e:
            print(f"Error in batch search: {e}")
            my_respones = {'result':[[{'document': {'contents': " \n "}}] for _ in queries ]} 
            import time
            time.sleep(1)
    return my_respones
    
def _passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        
        content = doc_item['document']['contents']
        # print("content:", content)
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


def batch_search(queries: List[str] = None) -> str:
    """
    Batchified search for queries.
    Args:
        queries: queries to call the search engine
    Returns:
        search results which is concatenated into a string
    """
    results = _batch_search(queries)['result']
    
    return [_passages2string(result) for result in results]



def execute_predictions(predictions: List[str], pad_token: str, active_mask=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        search_results = batch_search(search_queries)
        
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

def _process_next_obs(tokenizer, next_obs: List[str],max_obs_length) -> torch.Tensor:
    """Process next observations from environment."""
    
    next_obs_ids = tokenizer(
        next_obs, 
        padding='longest',
        return_tensors='pt',
        add_special_tokens=False,  # Prevents adding special tokens
    )['input_ids']

    if next_obs_ids.shape[1] > max_obs_length:
        print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
        print(f"-----[Debug]----- len(next_obs_ids): {next_obs_ids.shape[1]}")
        next_obs_ids = next_obs_ids[:, :max_obs_length]
        # assert 1==0
    return next_obs_ids