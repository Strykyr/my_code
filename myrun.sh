seq_len=100
folder_path=./data2/door_water/
#folder_path=./data2/door1/

if [ ! -d folder_path"logs" ]; then
    mkdir -p $folder_path"logs"
fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi

# python -u run.py \
#   --is_training 1 \
#   --model_id ETTh1_$seq_len'_'100 \
#   --model $model_name \
#   --features M \
#   --folder_path $folder_path \
#   --seq_len $seq_len \
#   --pred_len 10 \
#   --enc_in 2 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.005 >$folder_path'logs/'$model_name'.log'
# LSTM     'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            # 'DLinear': DLinear,
            # 'NLinear': NLinear,
            # 'Linear': Linear,

#NLinear

# for model_name in NLinear
# do
# python -u run.py \
#   --is_training 1 \
#   --model_id ETTh1_$seq_len'_'96 \
#   --model $model_name \
#   --folder_path $folder_path \
#   --features M \
#   --des 'Exp' \
#   --itr 1 --learning_rate 0.005 >$folder_path'logs/'$model_name'.log'

# done

# # # Autoformer Informer Transformer , Resnet, LSTM , Attention_LSTM
# for model_name in  Resnet_LSTM
# do
# python -u run.py \
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



# test 部分>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>..
# for model_name in Resnet_LSTM
# do
# python -u test.py \
#   --is_training 1 \
#   --model_id ETTh1_$seq_len'_'96 \
#   --model $model_name \
#   --folder_path $folder_path \
#   --features M \
#   --des 'Exp' \
#   --itr 1 --learning_rate 0.005 >$folder_path'logs/'$model_name'.log'

# done
# # LSTM Resnet
for model_name in Resnet_LSTM
do
python -u test.py \
  --is_training 1 \
  --model_id exchange_96_$pred_len \
  --model $model_name \
  --features M \
  --folder_path $folder_path \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --des 'Exp' \
  --itr 1 >$folder_path'logs/'$model_name'.log'
done