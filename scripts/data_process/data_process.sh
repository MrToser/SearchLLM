export PYTHONPATH=$PYTHONPATH:/home/avnet/xiongjing/sjh/agent/Search-R1
IDS_NAME=ids_augument
DIFFICULTY="filter_0" # easy,medium,hard
WORK_DIR=/home/avnet/xiongjing/sjh/agent/Search-R1
DIFFICULTY_UNDERSCORE="${DIFFICULTY//,/_}"
LOCAL_DIR="${WORK_DIR}/data/${IDS_NAME}_${DIFFICULTY_UNDERSCORE}"
IDS_DIR=$WORK_DIR/data/ids
export HIP_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
IDS_FILE="${IDS_DIR}/${IDS_NAME}.json"

## process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR --data_sources $DATA \
    --difficulty $DIFFICULTY --ids_file $IDS_FILE \
    

## process multiple dataset search format test file
# DATA=bamboogle,musique
# # DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
#     --local_dir $LOCAL_DIR --data_sources $DATA
