CHECKPOINT_DIR=/path/to/your/ckpt
RESUME_PATH=${CHECKPOINT_DIR}/output
mkdir -p ${RESUME_PATH}

STEPS=(12710)

GPU_ID=0
for step in "${STEPS[@]}"; do
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python llava/eval/eval_text.py \
            --text_data_path=data/QA_data/val_final.pkl \
            --checkpoint_path=${CHECKPOINT_DIR}/checkpoint-${step} \
            --output_dir=${RESUME_PATH}/checkpoint-${step} \
            --is_train=False &
        GPU_ID=$((GPU_ID+1))
done

wait
echo "Evaluation done!"
