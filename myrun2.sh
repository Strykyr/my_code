folder_path=./data2/door_water/
#folder_path=./data2/door1/

if [ ! -d folder_path"logs" ]; then
    mkdir -p $folder_path"logs"
fi

# # Linear NLinear DLinear LSTM CNN_LSTM
# --e_layers 2 \
#
    #  --e_layers 3 \
    #   --n_heads 16 \
    #   --d_model 128 \
    #   --d_ff 256 \
    #   --dropout 0.2\
    #   --fc_dropout 0.2\
    #   --head_dropout 0\
    #   --patch_len 16\
    #   --stride 8\


#   ============= 训练
# LSTM CNN_LSTM NLinear DLinear Linear Attention My_Model My_ModelD
for model_name in My_Model
do
python -u run2.py \
  --is_training 1 \
  --model_id exchange_96_$pred_len \
  --model $model_name \
  --features M \
  --folder_path $folder_path \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --fc_dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --factor 3 \
  --des 'Exp' \
  --itr 1 >$folder_path'logs/'$model_name'.log'
done


# # test new部分>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>..
# LSTM CNN_LSTM NLinear DLinear Linear Attention My_Model My_ModelD
for model_name in My_Model
do
python -u test2.py \
  --is_training 1 \
  --model_id exchange_96_$pred_len \
  --model $model_name \
  --features M \
  --folder_path $folder_path \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --fc_dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --factor 3 \
  --des 'Exp' \
  --itr 1 >$folder_path'logs/'$model_name'.log'
done





#==========================================================
# for model_name in My_Attention
# do
# python -u test2.py \
#   --is_training 1 \
#   --model_id exchange_96_$pred_len \
#   --model $model_name \
#   --features M \
#   --folder_path $folder_path \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --des 'Exp' \
#   --itr 1 >$folder_path'logs/'$model_name'.log'
# done