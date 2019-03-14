from pyspark import SparkConf, SparkContext 
from pyspark.streaming import StreamingContext 
from pyspark.streaming.kafka import KafkaUtils
from sklearn.externals import joblib
import math
import datetime
import pandas as pd


def dist(a,b): # 計算距離
    return math.sqrt((a*a)+(b*b))

def get_y_rotation(x,y,z): # 計算y軸角動量
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)

def get_x_rotation(x,y,z): # 計算x軸角動量
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)

def statusvote(df, clf): # 投票評分 
    status = ["靜止", "走路", "慢跑", "快跑"]
    lis = list(clf.predict(df))
    statuscount = [lis.count(0), lis.count(1), lis.count(2), lis.count(3)]
    statusscale = list(map(lambda var: round(var / len(lis), 3), statuscount))
    statusmode = statusscale.index(max(statusscale)) # 找出眾數
    return status[statusmode]

def calculate(string): # rdd.map function: 加入狀態
    target = string.split(',')
    target = [float(value) for value in target]
    # 算角動量
    accel_xyz = tuple(map(lambda var: float(var), target[4:]))
    accel_scaled_xyz = tuple(map(lambda var: var / 16384.0, accel_xyz))
    x_rotation = round(get_x_rotation(accel_scaled_xyz[0], accel_scaled_xyz[1], accel_scaled_xyz[2]), 4)
    y_rotation = round(get_y_rotation(accel_scaled_xyz[0], accel_scaled_xyz[1], accel_scaled_xyz[2]), 4)
    target.extend([x_rotation, y_rotation])
    # 裝到dataframe評分
    df = pd.DataFrame(columns=['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'rotation_x', 'rotation_y'])
    df = df.append(pd.Series(target[1:], index=['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'rotation_x', 'rotation_y']), ignore_index=True)
    ans = statusvote(df, clf)
    target.append((ans,1))
    target[0] = datetime.datetime.fromtimestamp(target[0]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]   
    return target

def vote(tuple1, tuple2): 
    ans = tuple1
    if tuple1[0] == "快跑" and tuple1[1] > 15: # 快跑條件預先判別
        ans = tuple1
    elif tuple2[0] == "快跑" and tuple2[1] > 15: 
        ans = tuple2
    elif tuple1[1] < tuple2[1]: # 其餘比較
        ans = tuple2
    return ans





if __name__ == '__main__':

    # 設定SparkStreaming參數
    conf = SparkConf().set('spark.io.compression.codec','snappy')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('warn')
    ssc = StreamingContext(sc,5)
	# 載入模型
    clf = joblib.load('KNN_8.pkl')
    # 設定Kafka參數
	zookeeper = '10.120.14.124:2181'
    topic = {"test":1} 
    groupid = 'consumer-group'
    kvs = KafkaUtils.createStream(ssc, zookeeper, groupid, topic)
    
	# Start to listen
    lines = kvs.repartition(1)
    ans = lines.map(lambda x: x[1]).map(calculate).map(lambda ls:ls[9]).reduceByKey(lambda x, y: x + y).reduce(vote)
    ans.pprint()
    ssc.start()  
    ssc.awaitTermination()