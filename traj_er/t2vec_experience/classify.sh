grid_size=(40 80 100 150 200 500 800)
source ~/py35env/bin/activate
cityname="singapore_taxi3"

for i in 6
do
    nohup python trajvec_classify.py -grid_size ${grid_size[$i]} -cityname $cityname -num_layers 1 > out/classify-$cityname-grid${grid_size[$i]}.out 2>&1 &

done