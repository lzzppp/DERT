
import csv
import glob
import random
from tqdm import tqdm
def random_index(rate):
    """随机变量的概率函数"""
    #
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))

    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index
image_dir_lists = glob.glob("bounding_box_test/0*.jpg")
image_dir_lists.extend(glob.glob("bounding_box_test/1*.jpg"))

person_data = {}
person_list = set()
for image_dir in tqdm(image_dir_lists, desc="生成人数集合"):
    person_list.add(image_dir[:24])
for person_ in tqdm(person_list, desc="人数据集初始化"):
    person_data[person_] = []
for image_dir in tqdm(image_dir_lists, desc="生成图像字典"):
    for person in person_list:
        if image_dir.startswith(person):
            person_data[person].append(image_dir)
id_ = 0
rate_matched = [8, 2]
f = open("test_list.csv", "w", encoding="utf-8", newline='')
csv_w = csv.writer(f, delimiter=',')
csv_w.writerow(['id', 'person1', 'person2', 'label'])
for per in tqdm(list(person_data.keys()), desc="生成匹配数据"):
    length = len(person_data[per])
    for i in range(length):
        for j in range(i + 1, length):
            is_produce = random_index(rate_matched)
            if is_produce:
                csv_w.writerow([id_, person_data[per][i], person_data[per][j], 1])
                id_ += 1

rate_unmatched = [999, 1]
LENGTH = len(list(person_data.keys()))
for k in tqdm(range(LENGTH), desc="生成不匹配数据"):
    for m in range(k + 1, LENGTH):
        per_k, per_m = list(person_data.keys())[k], list(person_data.keys())[m]
        length_k, length_m = len(person_data[per_k]), len(person_data[per_m])
        for i in range(length_k):
            for j in range(length_m):
                is_produce = random_index(rate_unmatched)
                if is_produce:
                    csv_w.writerow([id_, person_data[per_k][i], person_data[per_m][j], 0])
                    id_ += 1
