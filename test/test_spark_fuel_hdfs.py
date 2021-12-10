
from pyspark.sql import SparkSession
from sparkfuel.common.hdfs import Hdfs

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

hdfs = Hdfs(sc)
print(hdfs.exists('hdfs://ss-sng-dc-v2/data/SPARK/SNG/g_sng_qqkd_sng_buluo_kd_video_group/easyctr/sparkfuel/test/'))
