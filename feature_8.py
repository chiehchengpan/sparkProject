from pyspark import SparkConf, SparkContext 
from pyspark.streaming import StreamingContext 
from pyspark.streaming.kafka import KafkaUtils
from sklearn.externals import joblib
import math
import datetime
import pandas as pd


def dist(a,b):
    return math.sqrt((a*a)+(b*b))

def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)

def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)

def statusvote(df, clf):
    status = ["靜止", "走路", "慢跑", "快跑"]
    lis = list(clf.predict(df))
    statuscount = [lis.count(0), lis.count(1), lis.count(2), lis.count(3)]
    statusscale = list(map(lambda var: round(var / len(lis), 3), statuscount))
    statusmode = statusscale.index(max(statusscale))
    return status[statusmode]

def calculate(string):
    target = string.split(',')
    target = [float(value) for value in target]
    
    accel_xyz = tuple(map(lambda var: float(var), target[4:]))
    accel_scaled_xyz = tuple(map(lambda var: var / 16384.0, accel_xyz))
    x_rotation = round(get_x_rotation(accel_scaled_xyz[0], accel_scaled_xyz[1], accel_scaled_xyz[2]), 4)
    y_rotation = round(get_y_rotation(accel_scaled_xyz[0], accel_scaled_xyz[1], accel_scaled_xyz[2]), 4)
    target.extend([x_rotation, y_rotation])

    df = pd.DataFrame(columns=['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'rotation_x', 'rotation_y'])
    df = df.append(pd.Series(target[1:], index=['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'rotation_x', 'rotation_y']), ignore_index=True)
    ans = statusvote(df, clf)
    target.append((ans,1))
    target[0] = datetime.datetime.fromtimestamp(target[0]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
       
    return target




if __name__ == '__main__':
    
    conf = SparkConf().set('spark.io.compression.codec','snappy')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('warn')
    ssc = StreamingContext(sc,5)
    clf = joblib.load('KNN_8.pkl')
    zookeeper = '10.120.14.124:2181'
    topic = {"test":1} 
    groupid = 'consumer-group'
    kvs = KafkaUtils.createStream(ssc, zookeeper, groupid, topic)
    
    # lines = kvs.map(lambda x: x[1])
    # counts = lines.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
    # lines = lines.reduce(lambda a, b: a+'\n'+b)
    # lines = lines.reduce(lambda line1, line2: list(list(line1)).append(list(line2)))
    ccp = kvs.repartition(1)
    a = ccp.map(lambda x: x[1]).map(calculate).map(lambda ls:ls[9]).reduceByKey(lambda x, y: x + y)
    a.pprint()
    ssc.start()  
    ssc.awaitTermination()