export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PYTHONPATH:/home/avnet/xiongjing/sjh/agent/Search-R1
WORK_DIR=/home/avnet/xiongjing/sjh/agent/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_search_llm
export HIP_VISIBLE_DEVICES=7,1
# filter
DATA=nq,hotpotqa
# 5次一次yes则剔除
# python $WORK_DIR/scripts/data_process/filter_augument.py --local_dir $LOCAL_DIR --data_sources $DATA
# 根据 rollout剔除 存在无效action的样本/没有输出最终答案的样本/两次search内容存在一致性样本 -> 输出难度划分
python $WORK_DIR/scripts/data_process/second_filter_augument_and_difficulty_divide.py \
    --local_dir $LOCAL_DIR --data_sources $DATA

# merge

# ## process multiple dataset search format test file
# DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
