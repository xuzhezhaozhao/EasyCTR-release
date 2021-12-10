
from pyspark.sql import SparkSession
import sys
import os.path

if len(sys.argv) != 2:
    raise ValueError("Usage: <export_model_dir>")

export_model_dir = sys.argv[1]

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
hadoop_conf = sc._jsc.hadoopConfiguration()

p = sc._gateway.jvm.org.apache.hadoop.fs.Path(export_model_dir)
fs = p.getFileSystem(hadoop_conf)
files = fs.listStatus(p)

files = [f.getPath().toString() for f in files]
files = [f for f in files if os.path.basename(f).startswith('1')]
model_dir = sorted(files)[-1]

ts = os.path.basename(model_dir)
check_file = os.path.join(model_dir, ts + '.check')
p = sc._gateway.jvm.org.apache.hadoop.fs.Path(check_file)
if fs.createNewFile(p):
    print("Create check file '{}' successfully.".format(check_file))
else:
    raise ValueError("Create check file '{}' failed.".format(check_file))
