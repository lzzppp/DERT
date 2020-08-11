# for grid size experience 
vocab_size=(23668 18752 15816 10153 6977 1742 804)
grid_size=(40 80 100 150 200 500 800)
cityname="singapore_taxi3"
layer=1

for i in 6
do 
    nohup python t2vec.py -data experiment -vocab_size ${vocab_size[$i]} -checkpoint "saved_model/best_model_${cityname}_${grid_size[$i]}.0.pt" -mode 2 -encode_data "train" -grid_size ${grid_size[$i]} -cityname $cityname -num_layers $layer > out/encode-$cityname-train${grid_size[$i]}.out 2>&1 &
    export pid=$!
    echo $pid
    tail --pid=$pid -f /dev/null

    nohup python t2vec.py -data experiment -vocab_size ${vocab_size[$i]} -checkpoint "saved_model/best_model_${cityname}_${grid_size[$i]}.0.pt" -mode 2 -encode_data "test" -grid_size ${grid_size[$i]} -cityname $cityname -num_layers $layer > out/encode-$cityname-train${grid_size[$i]}.out 2>&1 &
    export pid=$!
    echo $pid
    tail --pid=$pid -f /dev/null
done