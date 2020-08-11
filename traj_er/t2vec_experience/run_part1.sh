region_name=region_geolife_len24.0
grid_size=180
train_num=6140
val_num=877

# cd /home/zhuzheng/data/data_preporcess/traj_split
# echo "generating small data..."
# python split_h5trian_test.py singaporetaxi_small

cd /home/lizepeng/t2vec_experience/preprocessing
echo "generate training data..."
julia preprocess.jl $grid_size $region_name $train_num $val_num

# echo "training model..."
