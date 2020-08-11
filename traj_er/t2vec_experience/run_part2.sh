vocab_size=11180
layer=3  # t2vec model layer(default in porto: 3)
cityname="geolife_pair24.0"
grid_size=180  # need to be same as part 1
region_name=region_geolife_len24.0
gpu_id=0

source ~/env/t2vecenv/bin/activate
echo "Training the model..."
python t2vec.py -data data -vocab_size $vocab_size -criterion_name "KLDIV" -knearestvocabs "data/$cityname-vocab-dist-cell$grid_size.h5" -cityname $cityname -grid_size $grid_size -num_layers $layer -gpu_id $gpu_id

echo "Transfer trajectory into grid sequence for encoding"
cd experiment
julia generate_traj.jl $grid_size $region_name

cd ..
echo "Encoding Trajectory"

python t2vec.py -data experiment -vocab_size $vocab_size -checkpoint "saved_model/best_model_${cityname}_${grid_size}.0.pt" -mode 2 -encode_data "train" -grid_size ${grid_size} -cityname $cityname -num_layers $layer -gpu_id $gpu_id

python t2vec.py -data experiment -vocab_size $vocab_size -checkpoint "saved_model/best_model_${cityname}_${grid_size}.0.pt" -mode 2 -encode_data "test" -grid_size ${grid_size} -cityname $cityname -num_layers $layer -gpu_id $gpu_id

# echo "Classify outcomes..."
# source ~/py35env/bin/activate
# python trajvec_classify.py -grid_size ${grid_size} -cityname $cityname -num_layers $layer -region_name $region_name 
