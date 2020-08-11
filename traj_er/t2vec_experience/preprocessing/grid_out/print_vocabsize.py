import os

grid_size = [40, 80, 100 ,150, 200, 500, 800]

vocab_size = ""
for s in grid_size:
    with open("region_singapore-grid%d.out" % s) as f:
        line = f.readlines()[-2]
        line = line.split(' ')
        vocab_size += ('%s ' % line[2])
print(vocab_size)
print(" ".join(map(str,grid_size)))