# Scala → Python: Apache Spark Master Conversion Guide
**DataFrame/Dataset + RDD, with Inputs & Outputs (interleaved)**

This guide gives you **Scala → Python** mappings for Spark with:
- **What it does**
- **Example Input**
- **Scala (DataFrame/Dataset)**
- **Python (PySpark)**
- **Example Output**
- **RDD Equivalent (Scala & Python)** right after each section, where it makes sense.

> Assumes you already created a `SparkSession spark`.  
> Outputs may differ in row order unless explicitly ordered.

---

## 0) Setup & SparkSession

**What it does:** Create the `SparkSession`, entry point for DataFrame/SQL.

**Scala**
```scala
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().appName("App").master("local[*]").getOrCreate()
import spark.implicits._
```

**Python**
```python
from pyspark.sql import SparkSession
spark = (SparkSession.builder.appName("App").master("local[*]").getOrCreate())
```

**RDD Equivalent:** (None — setup step)

---

## 1) Read / Write (CSV → Parquet)

**What it does:** Read CSV with header; write Parquet.

**Input (CSV)**
```csv
id,name,score
1,Amy,0.95
2,Bob,0.80
```

**Scala (DF)**
```scala
val df = spark.read.option("header","true").csv("s3://bucket/path/data.csv")
df.write.mode("overwrite").parquet("s3://bucket/path/out/")
df.show(false)
```

**Python (DF)**
```python
df = spark.read.option("header","true").csv("s3://bucket/path/data.csv")
(df.write.mode("overwrite").parquet("s3://bucket/path/out/"))
df.show(truncate=False)
```

**Output**
```
+---+----+-----+
|id |name|score|
+---+----+-----+
|1  |Amy |0.95 |
|2  |Bob |0.80 |
+---+----+-----+
```

**RDD Equivalent**
**Scala:**
```scala
val rdd = spark.sparkContext.textFile("s3://bucket/path/data.csv")
val header = rdd.first()
val rows = rdd.filter(_ != header).map(_.split(",")).map(a => (a(0), a(1), a(2)))
// Save
rows.map{case (id,n,s) => s"$id,$n,$s"}.saveAsTextFile("s3://bucket/path/out_rdd")
```

**Python:**
```python
rdd = spark.sparkContext.textFile("s3://bucket/path/data.csv")
header = rdd.first()
rows = (rdd.filter(lambda x: x != header)
           .map(lambda line: line.split(","))
           .map(lambda a: (a(0), a(1), a(2))))
rows.map(lambda t: ",".join(t)).saveAsTextFile("s3://bucket/path/out_rdd")
```

---

## 2) Schema Definition (read JSON with explicit types)

**What it does:** Apply a schema to JSON.

**Input (JSON lines)**
```json
{"id":1,"name":"Amy","ts":"2025-08-27T10:00:00Z"}
{"id":2,"name":"Bob","ts":"2025-08-27T10:05:00Z"}
```

**Scala (DF)**
```scala
import org.apache.spark.sql.types._
val schema = StructType(Seq(
  StructField("id", IntegerType, false),
  StructField("name", StringType, true),
  StructField("ts", TimestampType, true)
))
val df = spark.read.schema(schema).json("data.json")
df.printSchema(); df.show(false)
```

**Python (DF)**
```python
from pyspark.sql.types import *
schema = StructType([
  StructField("id", IntegerType(), False),
  StructField("name", StringType(), True),
  StructField("ts", TimestampType(), True),
])
df = spark.read.schema(schema).json("data.json")
df.printSchema(); df.show(truncate=False)
```

**Output (schema + rows)**
```
root
 |-- id: integer (nullable = false)
 |-- name: string (nullable = true)
 |-- ts: timestamp (nullable = true)

+---+----+-------------------+
|id |name|ts                 |
+---+----+-------------------+
|1  |Amy |2025-08-27 10:00:00|
|2  |Bob |2025-08-27 10:05:00|
+---+----+-------------------+
```

**RDD Equivalent**
**Scala:** parse JSON manually or with a library (less ergonomic than DF). Often you’d read JSON via DF and convert `df.rdd`.
```scala
val rdd = spark.read.schema(schema).json("data.json").rdd
```

**Python:** same idea:
```python
rdd = spark.read.schema(schema).json("data.json").rdd
```

---

## 3) Select / withColumn / when-otherwise

**What it does:** Rename, derive columns, conditional label.

**Input**
```
+---+----------+
|id |name      |
+---+----------+
|1  |Alexander |
|2  |Amy       |
+---+----------+
```

**Scala (DF)**
```scala
import org.apache.spark.sql.functions._
val df = Seq((1,"Alexander"),(2,"Amy")).toDF("id","name")
val out = df.select($"id", $"name".as("user_name"))
  .withColumn("name_len", length($"user_name"))
  .withColumn("tier", when($"name_len" > 5, lit("gold")).otherwise(lit("std")))
out.show(false)
```

**Python (DF)**
```python
from pyspark.sql.functions import col, length, when, lit
df = spark.createDataFrame([(1,"Alexander"),(2,"Amy")], ["id","name"])
out = (df.select(col("id"), col("name").alias("user_name"))
          .withColumn("name_len", length(col("user_name")))
          .withColumn("tier", when(col("name_len") > 5, lit("gold")).otherwise(lit("std"))))
out.show(truncate=False)
```

**Output**
```
+---+----------+--------+----+
|id |user_name |name_len|tier|
+---+----------+--------+----+
|1  |Alexander |9       |gold|
|2  |Amy       |3       |std |
+---+----------+--------+----+
```

**RDD Equivalent**
**Scala:**
```scala
val rdd = spark.sparkContext.parallelize(Seq((1,"Alexander"),(2,"Amy")))
val out = rdd.map{ case (id, name) =>
  val name_len = Option(name).map(_.length).getOrElse(0)
  val tier = if (name_len > 5) "gold" else "std"
  (id, name, name_len, tier)
}
out.collect().foreach(println)
```

**Python:**
```python
rdd = spark.sparkContext.parallelize([(1,"Alexander"),(2,"Amy")])
out = rdd.map(lambda t: (t[0], t[1], len(t[1]) if t[1] else 0,
                         "gold" if (len(t[1]) if t[1] else 0) > 5 else "std"))
print(out.collect())
```

---

## 4) Filtering / Ordering / Limiting

**What it does:** Filter rows, sort, limit.

**Input**
```
+----+------+-----+
|user|status|score|
+----+------+-----+
|u1  |ok    |0.93 |
|u2  |bad   |0.99 |
|u3  |ok    |0.88 |
|u4  |ok    |0.97 |
+----+------+-----+
```

**Scala (DF)**
```scala
import org.apache.spark.sql.functions._
val df = Seq(("u1","ok",0.93),("u2","bad",0.99),("u3","ok",0.88),("u4","ok",0.97))
  .toDF("user","status","score")
val out = df.filter($"status"==="ok" && $"score">0.9)
  .orderBy(desc("score")).limit(2)
out.show(false)
```

**Python (DF)**
```python
from pyspark.sql.functions import col
df = spark.createDataFrame([("u1","ok",0.93),("u2","bad",0.99),("u3","ok",0.88),("u4","ok",0.97)],
                           ["user","status","score"])
out = (df.filter((col("status")=="ok") & (col("score")>0.9))
         .orderBy(col("score").desc()).limit(2))
out.show(truncate=False)
```

**Output**
```
+----+------+-----+
|user|status|score|
+----+------+-----+
|u4  |ok    |0.97 |
|u1  |ok    |0.93 |
+----+------+-----+
```

**RDD Equivalent**
**Scala:**
```scala
val rdd = spark.sparkContext.parallelize(Seq(("u1","ok",0.93),("u2","bad",0.99),("u3","ok",0.88),("u4","ok",0.97)))
val out = rdd.filter{case (_,status,score) => status=="ok" && score>0.9}
             .sortBy(_._3, ascending=false)
             .take(2)
out.foreach(println)
```

**Python:**
```python
rdd = spark.sparkContext.parallelize([("u1","ok",0.93),("u2","bad",0.99),("u3","ok",0.88),("u4","ok",0.97)])
out = (rdd.filter(lambda t: t[1]=="ok" and t[2]>0.9)
          .sortBy(lambda t: t[2], ascending=False)
          .take(2))
print(out)
```

---

## 5) GroupBy / Aggregations

**What it does:** Group by key(s) and aggregate.

**Input**
```
+-------+-------+----------+
|country|user_id|latency_ms|
+-------+-------+----------+
|US     |a      |100       |
|US     |b      |120       |
|CA     |c      |200       |
|US     |a      |110       |
+-------+-------+----------+
```

**Scala (DF)**
```scala
import org.apache.spark.sql.functions._
val df = Seq(("US","a",100),("US","b",120),("CA","c",200),("US","a",110))
  .toDF("country","user_id","latency_ms")
val agg = df.groupBy($"country").agg(
  count("*").as("cnt"),
  avg($"latency_ms").as("avg_latency"),
  expr("approx_count_distinct(user_id)").as("unique_users")
).orderBy("country")
agg.show(false)
```

**Python (DF)**
```python
from pyspark.sql.functions import count, avg, expr, col
df = spark.createDataFrame([("US","a",100),("US","b",120),("CA","c",200),("US","a",110)],
                           ["country","user_id","latency_ms"])
agg = (df.groupBy(col("country"))
         .agg(count("*").alias("cnt"),
              avg(col("latency_ms")).alias("avg_latency"),
              expr("approx_count_distinct(user_id)").alias("unique_users"))
         .orderBy("country"))
agg.show(truncate=False)
```

**Output**
```
+-------+---+-----------+------------+
|country|cnt|avg_latency|unique_users|
+-------+---+-----------+------------+
|CA     |1  |200.0      |1           |
|US     |3  |110.0      |2           |
+-------+---+-----------+------------+
```

**RDD Equivalent**
**Scala (reduceByKey for sum, count, uniques via sets):**
```scala
val rdd = spark.sparkContext.parallelize(Seq(("US","a",100),("US","b",120),("CA","c",200),("US","a",110)))
val byCountry = rdd.map{case (c,u,l) => (c, (l, 1, Set(u)))}
  .reduceByKey{ case ((s1,c1,u1),(s2,c2,u2)) => (s1+s2, c1+c2, u1++u2) }
val out = byCountry.mapValues{case (sum, cnt, users) => (cnt, sum.toDouble/cnt, users.size)}
out.collect().foreach(println)
```

**Python:**
```python
rdd = spark.sparkContext.parallelize([("US","a",100),("US","b",120),("CA","c",200),("US","a",110)])
by_country = (rdd.map(lambda t: (t[0], (t[2], 1, {t[1]})))
                .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2] | b[2]))
                .mapValues(lambda v: (v[1], v[0]/v[1], len(v[2]))))
print(sorted(by_country.collect()))
```

---

## 6) Joins

**What it does:** Left outer join by key.

**Input**
```
left:  (id,name)  -> (1,Amy), (2,Bob), (3,Cal)
right: (id,dept)  -> (1,Sales), (2,Eng)
```

**Scala (DF)**
```scala
val left  = Seq((1,"Amy"),(2,"Bob"),(3,"Cal")).toDF("id","name")
val right = Seq((1,"Sales"),(2,"Eng")).toDF("id","dept")
val joined = left.join(right, Seq("id"), "left_outer").orderBy("id")
joined.show(false)
```

**Python (DF)**
```python
left  = spark.createDataFrame([(1,"Amy"),(2,"Bob"),(3,"Cal")], ["id","name"])
right = spark.createDataFrame([(1,"Sales"),(2,"Eng")], ["id","dept"])
joined = left.join(right, ["id"], "left_outer").orderBy("id")
joined.show(truncate=False)
```

**Output**
```
+---+----+-----+
|id |name|dept |
+---+----+-----+
|1  |Amy |Sales|
|2  |Bob |Eng  |
|3  |Cal |null |
+---+----+-----+
```

**RDD Equivalent**
**Scala:**
```scala
val rddL = spark.sparkContext.parallelize(Seq((1,"Amy"),(2,"Bob"),(3,"Cal")))
val rddR = spark.sparkContext.parallelize(Seq((1,"Sales"),(2,"Eng")))
val leftJoin = rddL.leftOuterJoin(rddR) // (id, (name, Option[dept]))
leftJoin.collect().foreach(println)
```

**Python:**
```python
rddL = spark.sparkContext.parallelize([(1,"Amy"),(2,"Bob"),(3,"Cal")])
rddR = spark.sparkContext.parallelize([(1,"Sales"),(2,"Eng")])
left_join = rddL.leftOuterJoin(rddR)
print(sorted(left_join.collect()))
```

---

## 7) Window Functions (lag, row_number)

**What it does:** Partition by device, order by time.

**Input**
```
(d1, "2025-08-27 10:00:00", 10.0)
(d1, "2025-08-27 10:01:00", 11.0)
(d1, "2025-08-27 10:02:00", 12.0)
```

**Scala (DF)**
```scala
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
val df = Seq(("d1","2025-08-27 10:00:00",10.0),
             ("d1","2025-08-27 10:01:00",11.0),
             ("d1","2025-08-27 10:02:00",12.0)).toDF("deviceId","eventTime","value")
val w = Window.partitionBy($"deviceId").orderBy($"eventTime")
val out = df.withColumn("lag_val", lag($"value",1).over(w))
            .withColumn("rn", row_number().over(w))
out.orderBy("eventTime").show(false)
```

**Python (DF)**
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, row_number, col
df = spark.createDataFrame([
  ("d1","2025-08-27 10:00:00",10.0),
  ("d1","2025-08-27 10:01:00",11.0),
  ("d1","2025-08-27 10:02:00",12.0),
], ["deviceId","eventTime","value"])
w = Window.partitionBy(col("deviceId")).orderBy(col("eventTime"))
out = (df.withColumn("lag_val", lag(col("value"),1).over(w))
         .withColumn("rn", row_number().over(w)))
out.orderBy("eventTime").show(truncate=False)
```

**Output**
```
+--------+-------------------+-----+-------+---+
|deviceId|eventTime          |value|lag_val|rn |
+--------+-------------------+-----+-------+---+
|d1      |2025-08-27 10:00:00|10.0 |null   |1  |
|d1      |2025-08-27 10:01:00|11.0 |10.0   |2  |
|d1      |2025-08-27 10:02:00|12.0 |11.0   |3  |
+--------+-------------------+-----+-------+---+
```

**RDD Equivalent:**
No native window functions in RDD API; you’d implement by key-grouping then sorting and manually computing lag/row numbers.

**Python sketch:**
```python
rdd = spark.sparkContext.parallelize([
  ("d1","2025-08-27 10:00:00",10.0),
  ("d1","2025-08-27 10:01:00",11.0),
  ("d1","2025-08-27 10:02:00",12.0),
])
by_key = rdd.groupBy(lambda t: t[0]).mapValues(list)
def add_lag(seq):
    seq = sorted(seq, key=lambda t: t[1])
    out = []
    for i,(dev, ts, v) in enumerate(seq):
        lagv = None if i==0 else seq[i-1][2]
        out.append((dev, ts, v, lagv, i+1))
    return out
print(by_key.flatMap(lambda kv: add_lag(kv[1])).collect())
```

---

## 8) Arrays / explode

**What it does:** Expand array elements into rows.

**Input**
```
(1, [{"k":1},{"k":2}])
```

**Scala (DF)**
```scala
import org.apache.spark.sql.functions._
val df = Seq( (1, Seq(Map("k"->1), Map("k"->2))) ).toDF("id","items")
val exploded = df.withColumn("item", explode($"items")).select($"id", $"item".getItem("k").as("k"))
exploded.show(false)
```

**Python (DF)**
```python
from pyspark.sql.functions import explode, col
df = spark.createDataFrame([(1, [{"k":1},{"k":2}])], ["id","items"])
exploded = (df.withColumn("item", explode(col("items")))
             .select(col("id"), col("item.k").alias("k")))
exploded.show(truncate=False)
```

**Output**
```
+---+---+
|id |k  |
+---+---+
|1  |1  |
|1  |2  |
+---+---+
```

**RDD Equivalent**
**Scala:**
```scala
val rdd = spark.sparkContext.parallelize(Seq((1, Seq(Map("k"->1), Map("k"->2)))))
val out = rdd.flatMap{ case (id, items) => items.map(_("k")).map(k => (id, k)) }
out.collect().foreach(println)
```

**Python:**
```python
rdd = spark.sparkContext.parallelize([(1, [{"k":1},{"k":2}])])
out = rdd.flatMap(lambda t: [(t[0], item["k"]) for item in t[1]])
print(out.collect())
```

---

## 9) UDFs

**What it does:** Normalize string (trim + lowercase).

**Input**
```
(1, "  Amy "), (2, null)
```

**Scala (DF)**
```scala
import org.apache.spark.sql.functions.udf
val norm = udf((s: String) => if (s==null) "" else s.trim.toLowerCase)
val df = Seq((1,"  Amy "), (2,null.asInstanceOf[String])).toDF("id","name")
val out = df.withColumn("norm_name", norm($"name"))
out.show(false)
```

**Python (DF)**
```python
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
@udf(StringType())
def norm(s):
    return "" if s is None else s.strip().lower()
df = spark.createDataFrame([(1,"  Amy "), (2,None)], ["id","name"])
out = df.withColumn("norm_name", norm(col("name")))
out.show(truncate=False)
```

**Output**
```
+---+------+---------+
|id |name  |norm_name|
+---+------+---------+
|1  |  Amy |amy      |
|2  |null  |         |
+---+------+---------+
```

**RDD Equivalent**
**Scala:**
```scala
val rdd = spark.sparkContext.parallelize(Seq((1,"  Amy "), (2,null.asInstanceOf[String])))
val out = rdd.map{ case (id, s) => (id, s, if (s==null) "" else s.trim.toLowerCase) }
out.collect().foreach(println)
```

**Python:**
```python
rdd = spark.sparkContext.parallelize([(1,"  Amy "), (2,None)])
out = rdd.map(lambda t: (t[0], t[1], "" if t[1] is None else t[1].strip().lower()))
print(out.collect())
```

---

## 10) Broadcast & Accumulators (broadcast filter)

**What it does:** Filter using a broadcast set.

**Input**
```
(id,country) -> (1,US),(2,MX),(3,CA)
```

**Scala (DF)**
```scala
import org.apache.spark.sql.functions._
val df = Seq((1,"US"),(2,"MX"),(3,"CA")).toDF("id","country")
val br = spark.sparkContext.broadcast(Set("US","CA"))
val out = df.filter($"country".isin(br.value.toSeq:_*))
out.show(false)
```

**Python (DF)**
```python
from pyspark.sql.functions import col
df = spark.createDataFrame([(1,"US"),(2,"MX"),(3,"CA")], ["id","country"])
br = spark.sparkContext.broadcast(set(["US","CA"]))
out = df.filter(col("country").isin(*br.value))
out.show(truncate=False)
```

**Output**
```
+---+-------+
|id |country|
+---+-------+
|1  |US     |
|3  |CA     |
+---+-------+
```

**RDD Equivalent**
**Scala:**
```scala
val rdd = spark.sparkContext.parallelize(Seq((1,"US"),(2,"MX"),(3,"CA")))
val br = spark.sparkContext.broadcast(Set("US","CA"))
val out = rdd.filter{ case (_,c) => br.value.contains(c) }
out.collect().foreach(println)
```

**Python:**
```python
rdd = spark.sparkContext.parallelize([(1,"US"),(2,"MX"),(3,"CA")])
br = spark.sparkContext.broadcast(set(["US","CA"]))
out = rdd.filter(lambda t: t[1] in br.value)
print(out.collect())
```

---

## 11) RDD Ops from DataFrame

**What it does:** Hop to RDD for pair counting.

**Input**
```
(id,val) -> (1,a),(1,b),(2,a)
```

**Scala (DF→RDD)**
```scala
val df = Seq((1,"a"),(1,"b"),(2,"a")).toDF("id","val")
val r2 = df.rdd.map(r => (r.getAs[Int]("id"), 1)).reduceByKey(_ + _)
r2.collect().foreach(println)
```

**Python (DF→RDD)**
```python
df = spark.createDataFrame([(1,"a"),(1,"b"),(2,"a")], ["id","val"])
r2 = df.rdd.map(lambda r: (r["id"], 1)).reduceByKey(lambda a,b: a+b)
print(sorted(r2.collect()))
```

**Output**
```
(1,2)
(2,1)
```
Pure RDD shown above.

---

## 12) SQL Registration

**What it does:** Query DF via SQL.

**Input**
```
(deviceId,temp) -> (d1,21.5),(d1,22.0),(d2,19.0)
```

**Scala (DF + SQL)**
```scala
val df = Seq(("d1",21.5),("d1",22.0),("d2",19.0)).toDF("deviceId","temp")
df.createOrReplaceTempView("events")
spark.sql("SELECT deviceId, COUNT(*) AS cnt FROM events GROUP BY deviceId")
  .orderBy("deviceId").show(false)
```

**Python (DF + SQL)**
```python
df = spark.createDataFrame([("d1",21.5),("d1",22.0),("d2",19.0)], ["deviceId","temp"])
df.createOrReplaceTempView("events")
spark.sql("SELECT deviceId, COUNT(*) AS cnt FROM events GROUP BY deviceId") \
     .orderBy("deviceId").show(truncate=False)
```

**Output**
```
+--------+---+
|deviceId|cnt|
+--------+---+
|d1      |2  |
|d2      |1  |
+--------+---+
```

**RDD Equivalent (group & count)**
```python
rdd = spark.sparkContext.parallelize([("d1",21.5),("d1",22.0),("d2",19.0)])
counts = rdd.map(lambda t: (t[0], 1)).reduceByKey(lambda a,b: a+b)
print(sorted(counts.collect()))
```

---

## 13) Structured Streaming: Kafka read + parse JSON

**What it does:** Read Kafka bytes, cast to string, parse JSON.

**Input (Kafka value)**
```json
{"deviceId":"d1","ts":"2025-08-27T10:00:00Z","latency_ms":100}
{"deviceId":"d2","ts":"2025-08-27T10:01:00Z","latency_ms":200}
```

**Scala (DF, streaming)**
```scala
import org.apache.spark.sql.functions._, org.apache.spark.sql.types._
val schema = new StructType()
  .add("deviceId","string").add("ts","string").add("latency_ms","integer")
val src = spark.readStream.format("kafka")
  .option("kafka.bootstrap.servers","broker:9092")
  .option("subscribe","topic-in")
  .option("startingOffsets","earliest").load()
val parsed = src.selectExpr("CAST(value AS STRING) AS json")
  .select(from_json($"json", schema).as("data")).select("data.*")
```

**Python (DF, streaming)**
```python
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType,StructField,StringType,IntegerType
schema = StructType([
  StructField("deviceId", StringType()),
  StructField("ts", StringType()),
  StructField("latency_ms", IntegerType())
])
src = (spark.readStream.format("kafka")
         .option("kafka.bootstrap.servers","broker:9092")
         .option("subscribe","topic-in")
         .option("startingOffsets","earliest").load())
parsed = (src.selectExpr("CAST(value AS STRING) AS json")
            .select(from_json(col("json"), schema).alias("data"))
            .select("data.*"))
```

**Output (Console sink sample)**
```
+--------+-------------------------+----------+
|deviceId|ts                       |latency_ms|
+--------+-------------------------+----------+
|d1      |2025-08-27T10:00:00Z     |100       |
|d2      |2025-08-27T10:01:00Z     |200       |
+--------+-------------------------+----------+
```

**RDD Equivalent:**
RDD API has no native Kafka source in modern Spark; use DF source then `parsed.rdd` if absolutely needed. DF/SQL is recommended for streaming.

---

## 14) Watermark + Window Aggregation (Event Time)

**What it does:** Event-time windowed avg with watermark.

**Scala (DF, streaming)**
```scala
import org.apache.spark.sql.functions._
val agg = parsed.withColumn("eventTime", $"ts".cast("timestamp"))
  .withWatermark("eventTime","10 minutes")
  .groupBy(window($"eventTime","5 minutes"), $"deviceId")
  .agg(avg($"latency_ms").as("avg_latency"))
```

**Python (DF, streaming)**
```python
from pyspark.sql.functions import col, avg, window
agg = (parsed.withColumn("eventTime", col("ts").cast("timestamp"))
       .withWatermark("eventTime","10 minutes")
       .groupBy(window(col("eventTime"),"5 minutes"), col("deviceId"))
       .agg(avg(col("latency_ms")).alias("avg_latency")))
```

**Output (Console sample)**
```
+--------------------------+--------+-----------+
|window                    |deviceId|avg_latency|
+--------------------------+--------+-----------+
|[2025-08-27 10:00,10:05]  |d1      |100.0      |
|[2025-08-27 10:00,10:05]  |d2      |200.0      |
+--------------------------+--------+-----------+
```

**RDD Equivalent:** Not available; use DF/SQL streaming.

---

## 15) Streaming Sinks & Triggers

**What it does:** Write streaming results to a sink with a trigger.

**Scala (DF)**
```scala
val q = agg.writeStream.format("console")
  .option("checkpointLocation","/chk/agg")
  .outputMode("update")
  .trigger(processingTime="30 seconds")
  .start()
q.processAllAvailable()
```

**Python (DF)**
```python
q = (agg.writeStream.format("console")
     .option("checkpointLocation","/chk/agg")
     .outputMode("update")
     .trigger(processingTime="30 seconds")
     .start())
q.processAllAvailable()
```

**Output:** periodic batches printed to console.

**RDD Equivalent:** Not applicable for modern structured streaming.

---

## 16) foreachBatch (JDBC Writes)

**What it does:** Per micro-batch insert/upsert to JDBC.

**Scala (DF)**
```scala
val q = agg.writeStream.foreachBatch { (batchDF, batchId) =>
  batchDF.write.format("jdbc")
    .option("url","jdbc:postgresql://host:5432/db")
    .option("dbtable","metrics")
    .option("user","user").option("password","secret")
    .mode("append").save()
}.option("checkpointLocation","/chk/jdbc")
 .outputMode("update").start()
```

**Python (DF)**
```python
q = (agg.writeStream.foreachBatch(lambda batchDF, batchId:
        batchDF.write.format("jdbc")
          .option("url","jdbc:postgresql://host:5432/db")
          .option("dbtable","metrics")
          .option("user","user").option("password","secret")
          .mode("append").save())
     .option("checkpointLocation","/chk/jdbc")
     .outputMode("update").start())
```

**RDD Equivalent:** Use DF writer; RDD lacks JDBC writer (you’d `mapPartitions` + driver libs, not recommended).

---

## 17) Deduplication with Watermark (Streaming)

**What it does:** Drop duplicates within watermark horizon.

**Input (example duplicates):**
```
(d1, e1, 2025-08-27T10:00:00Z)
(d1, e1, 2025-08-27T10:00:05Z)  // dup event_id
```

**Scala (DF)**
```scala
val deduped = parsed.withColumn("eventTime", $"ts".cast("timestamp"))
  .withWatermark("eventTime","10 minutes")
  .dropDuplicates("event_id","deviceId")
```

**Python (DF)**
```python
deduped = (parsed.withColumn("eventTime", col("ts").cast("timestamp"))
                 .withWatermark("eventTime","10 minutes")
                 .dropDuplicates(["event_id","deviceId"]))
```

**Output:** only first `(deviceId,event_id)` kept; later dup dropped.

**RDD Equivalent:** Not available for streaming semantics; implement batch dedupe with `reduceByKey` but no watermark semantics.

---

## 18) mapGroupsWithState / flatMapGroupsWithState (Stateful)

**What it does:** Per-key state across updates (e.g., running avg).

**Scala (DF/DS) – sketch**
```scala
import org.apache.spark.sql.streaming.{GroupState, GroupStateTimeout}
case class E(deviceId:String, ts:java.sql.Timestamp, value:Double)
case class Out(deviceId:String, avg:Double)
val ds = parsed.select($"deviceId",$"ts".cast("timestamp").as("ts"),$"value").as[E]
val out = ds.groupByKey(_.deviceId)
  .mapGroupsWithState(GroupStateTimeout.NoTimeout()){ (k, it, state) =>
    val (s,c) = it.foldLeft(state.getOption.getOrElse((0.0,0))){ case ((sum,cnt), e) => (sum+e.value,cnt+1) }
    val avg = s / c
    state.update((s,c))
    Out(k, avg)
  }
```

**Python (DF) – sketch**
```python
from pyspark.sql.streaming import GroupState, GroupStateTimeout
def fn(key, rows, state: GroupState):
    s, c = state.get() if state.exists else (0.0, 0)
    for r in rows:
        s += float(r.value); c += 1
    avg = s / c if c else 0.0
    state.update((s,c))
    yield (key, avg)

out = (parsed.groupByKey(lambda r: r.deviceId)
             .flatMapGroupsWithState(outputMode="update",
                                     timeoutConf=GroupStateTimeout.NoTimeout,
                                     func=fn))
```

**RDD Equivalent:** No direct streaming state op; roll your own with external storage/checkpoints (complex). Prefer DF.

---

## 19) Datasets (Scala) vs DataFrames (Python)

**What it does:** Show typed vs untyped.

**Scala**
```scala
case class User(id:Int, name:String)
val ds = Seq(User(1,"Amy")).toDS()
ds.map(u => u.name.toUpperCase).show()
```

**Python**
```python
df = spark.createDataFrame([(1,"Amy")], ["id","name"])
df.selectExpr("upper(name) as NAME").show()
```

**Output**
**Scala:**
```
+---+
|AMY|
+---+
```
**Python:**
```
+----+
|NAME|
+----+
|AMY |
+----+
```
**RDD Equivalent:** Both APIs can convert to RDD; no typing guarantees in RDD.

---

## 20) Porting Gotchas (illustrative)

**What it does:** Common pitfalls & better equivalents.
- Prefer built-ins or pandas_udf over Python UDFs.
- Avoid `collect()` on large data.
- Match streaming output modes & watermarks exactly.
- Replace Scala case classes/encoders with explicit schemas in Python.

**Example**
```scala
// Scala (good)
df.withColumn("clean", lower(trim($"raw")))
```
```python
# Python (good)
from pyspark.sql.functions import lower, trim, col
df.withColumn("clean", lower(trim(col("raw"))))
```

**RDD Equivalent:** Apply transform functions in map, but you lose Catalyst optimizations.

---

## Part 2 — Core RDD Ops (Quick Reference)

Already shown interleaved, but here’s a compact list with IO:
- `parallelize`, `map`, `filter`
- **(K,V) ops:** `mapToPair`, `reduceByKey`, `groupByKey`, `join`, `leftOuterJoin`
- `flatMap` (tokenize)
- **Actions:** `collect`, `count`, `first`, `take`, `saveAsTextFile`
- **Sort:** `sortBy`, `sortByKey`
- **Partitions:** `repartition`, `coalesce`
- **Pair aggregations:** `aggregateByKey`, `combineByKey` (advanced rollups)

### Classic Word Count
**Scala**
```scala
val lines = spark.sparkContext.parallelize(Seq("to be or not", "to be"))
val counts = lines.flatMap(_.split("\\s+")).map((_,1)).reduceByKey(_+_)
counts.collect().foreach(println)
```

**Python**
```python
lines = spark.sparkContext.parallelize(["to be or not", "to be"])
counts = (lines.flatMap(lambda s: s.split())
               .map(lambda w: (w,1))
               .reduceByKey(lambda a,b: a+b))
print(sorted(counts.collect()))
```

**Output**
```
(be,2), (not,1), (or,1), (to,2)
```

---

## Appendix: Minimal creators used above

**Scala**
```scala
import spark.implicits._
```

**Python**
```python
# spark.createDataFrame([...], ["col1","col2"])
```
