from pyspark import SparkConf, SparkContext 
from pyspark.streaming import StreamingContext 
from pyspark.streaming.kafka import KafkaUtils
from sklearn.externals import joblib
import math
import datetime
import pandas as pd
import numpy as np
import mysql.connector 

def dist(a,b):
    return math.sqrt((a*a)+(b*b))

def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)

def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)

def transformToDate(ls):
    if len(ls) == 9:
       ls[0] = datetime.datetime.fromtimestamp(ls[0]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
       return ls
    else:
       return ls

def calculateRotation(ls):
    if len(ls) == 7:
       ls = [float(value) for value in ls]
       accel_xyz = tuple(map(lambda var: float(var), ls[4:]))
       accel_scaled_xyz = tuple(map(lambda var: var / 16384.0, accel_xyz))
       x_rotation = round(get_x_rotation(accel_scaled_xyz[0], accel_scaled_xyz[1], accel_scaled_xyz[2]), 4)
       y_rotation = round(get_y_rotation(accel_scaled_xyz[0], accel_scaled_xyz[1], accel_scaled_xyz[2]), 4)
       ls.extend([x_rotation, y_rotation])
       return ls
def statusvote(df, clf):
    status = ["靜止", "走路", "慢跑", "快跑"]
    lis = list(clf.predict(df))
    statuscount = [lis.count(0), lis.count(1), lis.count(2), lis.count(3)]
    statusscale = list(map(lambda var: round(var / len(lis), 3), statuscount))
    statusmode = statusscale.index(max(statusscale))
    return status[statusmode]

def predictStatus(ls):
    column_names = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'rotation_x', 'rotation_y',
                    'std_gyro_x', 'std_gyro_y', 'std_gyro_z', 'std_accel_x', 'std_accel_y', 'std_accel_z', 'std_rot_x', 'std_rot_y']
    df = pd.DataFrame(columns = column_names)
    df = df.append(pd.Series(ls, index=column_names), ignore_index=True)
    ans = statusvote(df, clf)
    ls.append(ans)
    return ls
    
def stdevCal(ls):
    if isinstance(ls[0], str):
       ls.append(ls[-1])
    else:
       ls.append(round(np.std(ls),3))
    return ls

def sortBy(rdd):
    return rdd.sortByKey()

def addStatus(ls):
    dic = ["靜止", "走路", "慢跑", "快跑"]
    ls.append(dic[int(np.random.randint(4))])
    return ls

def pickLastTwo(ls):
    target = []
    target.append(ls[-2])
    target.append(ls[-1])
    return target

def voteForAns(ls):
    status = ["靜止", "走路", "慢跑", "快跑"]
    if ls[0] in status:

        statuscount = [ls.count(status[0]), ls.count(status[1]), ls.count(status[2]), ls.count(status[3])]
        ans = statuscount.index(max(statuscount))
        return [status[ans]]*len(ls)
    else:
        return ls
def filterStatus(tp):
    if tp[1][0] in ["靜止", "走路", "慢跑", "快跑"]:
       return True
    else:
       return False
def voteForRedis(ls):
    status = ["靜止", "走路", "慢跑", "快跑"]
    statuscount = [ls.count(status[0]), ls.count(status[1]), ls.count(status[2]), ls.count(status[3])]
    ans = statuscount.index(max(statuscount))
    return status[ans]

def changeBeforeVote(tp):
    time = tp[0]
    ls = tp[1]
    ans_ls = []
    ans_ls.append((0 , time))
    for i in range(len(ls)-1):
        ans_ls.append(((i+1), ls[i]))
    ans_ls.append((len(ls), ls[len(ls)-1]))
    return ans_ls

def changeAfterVote(tp):
    ls = tp[1]
    ans_ls = []
    for i in range(len(ls)):
        ans_ls.append(((i+1), ls[i]))
    return ans_ls

def timeAsKey(tp):
    ls = tp[1]
    return (ls[0], ls[1:])

def redisUpdate(tp):
    if len(tp[1]) == 1:
       return True
    if len(tp[1]) == 4:
       if tp[1][3] == tp[1][2] and tp[1][2] == tp[1][1]:
          if tp[1][3] != tp[1][0]:
             return True
          else: return False
       else: return False
    else: return False



if __name__ == '__main__':
    
    # 設定Spark Streaming連線參數
    conf = SparkConf().set('spark.io.compression.codec','snappy')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('warn')
    ssc = StreamingContext(sc, 1)
    
    # 載入模型
    clf = joblib.load('KNN_16.pkl')
    
    # Kafka 連線參數
    zookeeper = '10.120.14.124:2181'
    topic = {"test":1} 
    groupid = 'consumer-group'
    Dstream = KafkaUtils.createStream(ssc, zookeeper, groupid, topic)
    
    # 開始收取streamming data
 
    # 計算rotation & 把時間轉為字串格式
    row_with_rotation = Dstream.window(10, 1).map(lambda tp: tp[1].split(',')).map(calculateRotation).map(transformToDate)
    
    # 用(key,value)轉為column-base rdd, 並計算標準差
    column_base = row_with_rotation.flatMap(lambda ls: [('9. time', ls[0]), ('1. gyro_x', ls[1]), ('2. gyro_y', ls[2]),('3. gyro_z', ls[3]), ('4. accel_x', ls[4]), 
                                                                            ('5. accel_y', ls[5]), ('6. accel_z', ls[6]),('7. rot_x', ls[7]), ('8. rot_y', ls[8])])\
                     .groupByKey().mapValues(list).transform(sortBy)
    column_base_with_stdev = column_base.mapValues(stdevCal)
    


    row_with_stdev = column_base_with_stdev.mapValues(pickLastTwo).map(lambda x: x[1])\
          .flatMap(lambda ls:[('1. now', ls[0]), ('2. stdev', ls[1])])\
          .groupByKey().mapValues(list).transform(sortBy)
    
    row_with_stdev = row_with_stdev.map(lambda tp: (tp[1][-1], tp[1][:-1])).reduceByKey(lambda ls1, ls2: ls1 + ls2) 
    single_record = row_with_stdev.mapValues(predictStatus)
    
    window_records_before_vote = single_record.window(10, 10).transform(sortBy)
    vote = window_records_before_vote.flatMap(changeBeforeVote).groupByKey().mapValues(list).transform(sortBy)
    window_records_after_vote  = vote.mapValues(voteForAns).flatMap(changeAfterVote).groupByKey().mapValues(list).transform(sortBy).map(timeAsKey)

    window_state_after_vote = vote.filter(filterStatus).mapValues(voteForRedis).map(lambda tp:tp[1])
    window_state_list = window_state_after_vote.window(40, 10).map(lambda string: ('every_5s_status',string)).groupByKey().mapValues(list)
    update_state = window_state_list.filter(redisUpdate).map(lambda tp: ('Update_status',tp[1][-1]))

  
    # Dstream.pprint()
    # row_with_rotation.map(lambda tp: tp[1:]).pprint()
    # column_base.pprint()
    # column_base_with_stdev.pprint()
    # row_with_stdev.pprint()
    single_record.map(lambda tp: tp[1]).pprint()
    # window_records_before_vote.pprint(10)
    # window_records_after_vote.pprint(10)
    window_state_list.pprint()
    update_state.pprint()
      
    
    ssc.start()  
    ssc.awaitTermination()