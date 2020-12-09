
import csv

if __name__ == '__main__':
    file_list = ['train.csv', 'valid.csv', 'test.csv']
    for file in file_list:
        writer = csv.writer(open('ag_value_dataset/' + file, 'w', newline=''))
        reader = csv.reader(open('ag_dataset/' + file))
        next(reader)
        # name = ['id', 'label', 'left_title', 'left_category', 'left_brand', 'left_modelno', 'left_price', 'left_value', 'right_title', 'right_category', 'right_brand', 'right_modelno', 'right_price', 'right_value']
        name = ['id', 'label', 'left_title', 'left_manufacturer', 'left_price', 'left_value', 'right_title', 'right_manufacturer', 'right_price', 'right_value']
        writer.writerow(name)
        for line in reader:
            # writer.writerow(line[:7] + [' '.join(line[2:7])] + line[7:] + [' '.join(line[7:])])
            writer.writerow(line[:5] + [' '.join(line[2:5])] + line[5:] + [' '.join(line[5:])])