#!/bin/bash
BASE_PORT=31110
BASE_TM_PORT=50110
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/construction_split #! evaluate routes file
TEAM_AGENT=leaderboard/team_code/reasonplan_b2d_agent.py
TEAM_CONFIG=/path/to/your/checkpoint #! llava checkpoint
BASE_CHECKPOINT_ENDPOINT=eval
SAVE_PATH=eval_output/  #! results will be saved in this folder
PLANNER_TYPE=only_traj

mkdir -p $SAVE_PATH

GPU_RANK=0  #! GPU rank
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${SAVE_PATH}eval.json"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${SAVE_PATH}debug_results.log
