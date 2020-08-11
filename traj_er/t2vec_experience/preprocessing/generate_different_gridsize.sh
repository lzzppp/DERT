size_array=( 40 80 100 150 200 500 800 )
cityname=region_singapore

for i in "${size_array[@]}"
do 
    nohup julia preprocess.jl $i $cityname > grid_out/$cityname-grid$i.out 2>&1 &
done