layer=1  # t2vec model layer(default in porto: 3)
cityname="geolife_top100"
grid_size=180  # need to be same as part 1
region_name=region_geolife_top100
gpu_id=0


python trajvec_classify.py -grid_size ${grid_size}  -num_layers $layer -region_name $region_name 