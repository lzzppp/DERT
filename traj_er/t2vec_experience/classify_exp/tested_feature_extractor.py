import numpy as np
import h5py
from datetime import datetime
from geopy.distance import distance 
import argparse
import pickle 
import json
import os

class TestedFeatureExtractor:
    driving_time_norm = 1
    def __init__(self, selected_feature, norm_param):
        self.selected_feature = selected_feature
        self._set_norm_param(norm_param)

    def _set_norm_param(self, norm_param):
        self.norm_driving_time = norm_param["driving_time"]
        self.norm_driving_distance = norm_param["driving_distance"]
        self.norm_speed = norm_param["speed"]

    # def _set_param_dims(self):
    #     dim = 0
    #     self.norm_feature_dim = {}
    #     if 'time_of_day' in self.selected_feature:
    #         dim += 12
    #     if 'day_week' in self.selected_feature:
    #         dim += 7
    #     for feature in ['trip_time']:

    def extract_from_h5(self, h5_path, save_path, number='all'):
        f = h5py.File(h5_path, 'r')
        traj_nums = f.attrs['traj_nums']
        func = self.spatial_temporal_features_func(f)
        if number == 'all':
            out = np.array(list(map(func, range(traj_nums))))
        elif isinstance(number, int):
            out = np.array(list(map(func, range(number))))
        else:
            raise Exception("number of needed trajectories should be set properly")
        f.close()
        np.save(save_path, out)

    
    def spatial_temporal_features_func(self, f):
        def norma_traj(point):
            """If traj point in normal range. (lon, lat)"""
            return point[0] >= -180 and point[0] <= 180 and point[1] >= -90 and point[1] <= 90

        def func(tid):
            trajs = np.array(f['trips/%d' % tid])
            times = np.array(f['timestamps/%d' % tid])
            trajs = np.array(list(filter(norma_traj, trajs)))
        
            # time of day, day of week
            out_feature = []
            if 'time_of_day' in self.selected_feature or 'day_of_week' in self.selected_feature:
                day_hour, date_week = self.unix_to_weekday_and_hour(times[0])
                if 'time_of_day' in self.selected_feature:
                    out_feature += self.one_hot(day_hour, 12)
                if 'day_of_week' in self.selected_feature:
                    out_feature += self.one_hot(date_week, 7)
            
            if 'trip_time' in self.selected_feature:
                out_feature.append(self._normalize(times[-1] - times[0], self.norm_driving_time))

            if 'avg_speed' in self.selected_feature or 'max_speed' in self.selected_feature or 'drving_time' in self.selected_feature:
                out_feature += self.driving_feature(trajs, times, len(trajs) == 0)
            return out_feature

        return func 

    
    def driving_feature(self, trips, times, abnormal=False):
        
        distances = [coord_distance(coords) for coords in zip(trips[1:], trips[:-1])]
        seg_times = times[1:] - times[:-1]
        speeds = [distances[i] / seg_times[i] if seg_times[i] != 0.0 else 0.0 for i in range(len(distances))]
        out_feature = []
        if 'driving_distance' in self.selected_feature:
            if not abnormal:
                out_feature.append(self._normalize(sum(distances), self.norm_driving_distance))
            else:
                out_feature.append(0.0)
        if 'avg_speed' in self.selected_feature:
            if not abnormal:
                out_feature.append(self._normalize(np.mean(speeds), self.norm_speed))
            else:
                out_feature.append(0.0)
        if 'max_speed' in self.selected_feature:
            if not abnormal:
                out_feature.append(self._normalize(np.max(speeds), self.norm_speed))
            else:
                out_feature.append(0.0)

        return out_feature

    def unix_to_weekday_and_hour(self, unix_time):
        """Get hour and day of the week
        
        For hour of day, it will be divided to 12 parts
        Return:
        [day_part, day_of_week]
        """
        date = datetime.fromtimestamp(unix_time)
        return [date.hour // 2, date.weekday()] 

    def one_hot(self, id, len):
        return [0 if i != id else 1 for i in range(len)]

    def _normalize(self, value, max_value):
        # In this case, all value will be above 0, and I want to normalize them to -1,1
        # normalize all column at once will be more efficient
        if value > max_value:
            return 1.0
        else:
            return value / max_value * 2 - 1.0

def coord_distance(coords):
    """return distance between two points
    
    geopy.distance.distance accept [lat, lon] input, while this dataset is [lon, lat]
    """
    return distance((coords[0][1], coords[0][0]), (coords[1][1], coords[1][0])).meters

def get_saved_path(city_name, train_or_test):
    data_root = '/data3/zhuzheng/trajecotry/feature'
    return os.path.join(data_root, city_name + '_' + train_or_test)

parser = argparse.ArgumentParser(description="extral trajectory's temporal related feature")
parser.add_argument("-region_name", type=str, default="region_porto_top100", help="")

args = parser.parse_args()

if __name__ == "__main__":

    selected_feature = ['time_of_day', 'day_of_week', 'avg_speed', 'max_speed', 'trip_distance', 'trip_time']

    with open('../hyper-parameters.json', 'r') as f:
        hyper_param = json.loads(f.read())

    with open('normalize_param.json', 'r') as f:
        norm_param = json.loads(f.read())
    feature_extractor = TestedFeatureExtractor(selected_feature, norm_param[args.region_name])
    train_h5_path = hyper_param[args.region_name]['filepath']
    test_h5_path = hyper_param[args.region_name]['testpath']

    feature_extractor.extract_from_h5(train_h5_path, get_saved_path(hyper_param[args.region_name]['cityname'], 'train'))
    feature_extractor.extract_from_h5(test_h5_path, get_saved_path(hyper_param[args.region_name]['cityname'], 'test'))

    