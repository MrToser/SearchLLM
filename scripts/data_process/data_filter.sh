export PYTHONPATH=$PYTHONPATH:/home/avnet/xiongjing/sjh/agent/Search-R1
WORK_DIR=/home/avnet/xiongjing/sjh/agent/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_search_llm
export HIP_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
# filter
DATA=nq,hotpotqa
# 回答yes 则剔除
python $WORK_DIR/scripts/data_process/filter.py --local_dir $LOCAL_DIR --data_sources $DATA
# 难易划分
python $WORK_DIR/scripts/data_process/difficulty_divide.py --local_dir $LOCAL_DIR --data_sources $DATA

# merge

# ## process multiple dataset search format test file
# DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
