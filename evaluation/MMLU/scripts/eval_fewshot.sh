name=OpenBT5-5shot
input_folder="./data/5shot"
current_model_path="/public/home/ljt/LLM/wpz/hf_models/OpenBT5-Flan"
current_template="make_ABCD_input_5_shot"
current_output_folder="./output/${name}"
log_name="logs/${name}"
max_length=1024
decoder_max_length=256

export CUDA_VISIBLE_DEVICES=2
nohup python -u main.py \
    --model-path $current_model_path \
    --max-length $max_length \
    --input-folder $input_folder \
    --output-folder $current_output_folder \
    --template-type $current_template \
    --decoder-max-length $decoder_max_length \
    --add-prefix \
    --ptoken S > $log_name 2>&1 &
