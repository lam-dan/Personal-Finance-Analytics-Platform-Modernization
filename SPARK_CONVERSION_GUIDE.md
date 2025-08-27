# Scala → Python: Apache Spark Conversion Guide (with Inputs & Outputs)

Quick, side-by-side mappings from **Scala (Dataset/DataFrame API)** to **PySpark**, each with:
- **What it does**
- **Example Input** (small, tailored)
- **Scala code**
- **Python code**
- **Example Output** (as if you `show(false)` / `collect()`)

> All examples assume an existing `SparkSession spark`.  
> Outputs may differ in row order unless explicitly ordered.

---

## 0) Setup & SparkSession

**What it does:** Create the `SparkSession`, entry point for DataFrame/SQL.

**Input:** *(N/A — setup)*

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

**Output:** (N/A — a Spark UI and session)

---

## 1) Read / Write (CSV → Parquet)
**What it does:** Reads CSV with header; writes Parquet.

**Input (CSV sample):**
```csv
id,name,score
1,Amy,0.95
2,Bob,0.80
```

**Scala**
```scala
val df = spark.read.option("header","true").csv("s3://bucket/path/data.csv")
df.write.mode("overwrite").parquet("s3://bucket/path/out/")
df.show(false)
```

**Python**
```python
df = (spark.read.option("header","true").csv("s3://bucket/path/data.csv"))
(df.write.mode("overwrite").parquet("s3://bucket/path/out/"))
df.show(truncate=False)
```

**Output**
```sql
+---+----+-----+
|id |name|score|
+---+----+-----+
|1  |Amy |0.95 |
|2  |Bob |0.80 |
+---+----+-----+
```

---

## 2) Schema Definition (read JSON with explicit types)
**What it does:** Applies a schema to parsed JSON.

**Input (JSON lines):**
```json
{"id":1,"name":"Amy","ts":"2025-08-27T10:00:00Z"}
{"id":2,"name":"Bob","ts":"2025-08-27T10:05:00Z"}
```

**Scala**
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

**Python**
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

+---+----+-----------------------+
|id |name|ts                     |
+---+----+-----------------------+
|1  |Amy |2025-08-27 10:00:00    |
|2  |Bob |2025-08-27 10:05:00    |
+---+----+-----------------------+
```

---

## 3) Select / withColumn / when-otherwise
**What it does:** Rename, derive columns, conditional label.

**Input (DF):**
```sql
+---+-----+
|id |name |
+---+-----+
|1  |Alexander|
|2  |Amy  |
+---+-----+
```

**Scala**
```scala
import org.apache.spark.sql.functions._
val out = spark.createDataFrame(Seq((1,"Alexander"),(2,"Amy"))).toDF("id","name")
  .select($"id", $"name".as("user_name"))
  .withColumn("name_len", length($"user_name"))
  .withColumn("tier", when($"name_len" > 5, lit("gold")).otherwise(lit("std")))
out.show(false)
```

**Python**
```python
from pyspark.sql.functions import col, length, when, lit
inp = spark.createDataFrame([(1,"Alexander"),(2,"Amy")], ["id","name"])
out = (inp
       .select(col("id"), col("name").alias("user_name"))
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

---

## 4) Filtering / Ordering / Limiting
**What it does:** Keep rows matching criteria, sort, limit.

**Input**
```
+----+-----+-----+
|user|status|score|
+----+-----+-----+
|u1  |ok   |0.93 |
|u2  |bad  |0.99 |
|u3  |ok   |0.88 |
|u4  |ok   |0.97 |
+----+-----+-----+
```

**Scala**
```scala
import org.apache.spark.sql.functions._
val df = Seq(("u1","ok",0.93),("u2","bad",0.99),("u3","ok",0.88),("u4","ok",0.97))
  .toDF("user","status","score")
val out = df.filter($"status"==="ok" && $"score">0.9)
  .orderBy(desc("score")).limit(2)
out.show(false)
```

**Python**
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

---

## 5) GroupBy / Aggregations
**What it does:** Group and compute multiple aggregates.

**Input**
```
+-------+----------+----------+
|country|user_id   |latency_ms|
+-------+----------+----------+
|US     |a         |100       |
|US     |b         |120       |
|CA     |c         |200       |
|US     |a         |110       |
+-------+----------+----------+
```

**Scala**
```scala
import org.apache.spark.sql.functions._
val df = Seq(("US","a",100),("US","b",120),("CA","c",200),("US","a",110))
  .toDF("country","user_id","latency_ms")
val agg = df.groupBy($"country").agg(
  count("*").as("cnt"),
  avg($"latency_ms").as("avg_latency"),
  expr("approx_count_distinct(user_id)").as("unique_users")
)
agg.orderBy($"country").show(false)
```

**Python**
```python
from pyspark.sql.functions import count, avg, expr, col
df = spark.createDataFrame([("US","a",100),("US","b",120),("CA","c",200),("US","a",110)],
                           ["country","user_id","latency_ms"])
agg = (df.groupBy(col("country"))
         .agg(count("*").alias("cnt"),
              avg(col("latency_ms")).alias("avg_latency"),
              expr("approx_count_distinct(user_id)").alias("unique_users")))
agg.orderBy("country").show(truncate=False)
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

---

## 6) Joins
**What it does:** Left outer join on id.

**Input**
```sql
left:
+---+-----+
|id |name |
+---+-----+
|1  |Amy  |
|2  |Bob  |
|3  |Cal  |
+---+-----+

right:
+---+------+
|id |dept  |
+---+------+
|1  |Sales |
|2  |Eng   |
+---+------+
```

**Scala**
```scala
val left  = Seq((1,"Amy"),(2,"Bob"),(3,"Cal")).toDF("id","name")
val right = Seq((1,"Sales"),(2,"Eng")).toDF("id","dept")
val joined = left.join(right, Seq("id"), "left_outer").orderBy("id")
joined.show(false)
```

**Python**
```python
left  = spark.createDataFrame([(1,"Amy"),(2,"Bob"),(3,"Cal")], ["id","name"])
right = spark.createDataFrame([(1,"Sales"),(2,"Eng")], ["id","dept"])
joined = left.join(right, ["id"], "left_outer").orderBy("id")
joined.show(truncate=False)
```

**Output**
```sql
+---+----+-----+
|id |name|dept |
+---+----+-----+
|1  |Amy |Sales|
|2  |Bob |Eng  |
|3  |Cal |null |
+---+----+-----+
```

---

## 7) Window Functions (lag, row_number)
**What it does:** Partition by device, order by time.

**Input**
```
+--------+-------------------+-----+
|deviceId|eventTime          |value|
+--------+-------------------+-----+
|d1      |2025-08-27 10:00:00|10.0 |
|d1      |2025-08-27 10:01:00|11.0 |
|d1      |2025-08-27 10:02:00|12.0 |
+--------+-------------------+-----+
```

**Scala**
```scala
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
val df = Seq(
  ("d1","2025-08-27 10:00:00",10.0),
  ("d1","2025-08-27 10:01:00",11.0),
  ("d1","2025-08-27 10:02:00",12.0)
).toDF("deviceId","eventTime","value")
val w = Window.partitionBy($"deviceId").orderBy($"eventTime")
val out = df.withColumn("lag_val", lag($"value",1).over(w))
            .withColumn("rn", row_number().over(w))
out.orderBy("eventTime").show(false)
```

**Python**
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
```sql
+--------+-------------------+-----+-------+---+
|deviceId|eventTime          |value|lag_val|rn |
+--------+-------------------+-----+-------+---+
|d1      |2025-08-27 10:00:00|10.0 |null   |1  |
|d1      |2025-08-27 10:01:00|11.0 |10.0   |2  |
|d1      |2025-08-27 10:02:00|12.0 |11.0   |3  |
+--------+-------------------+-----+-------+---+
```

---

## 8) Arrays / explode
**What it does:** Explodes array column into multiple rows.

**Input**
```
+---+-------------+
|id |items        |
+---+-------------+
|1  |[{"k":1},{"k":2}]|
+---+-------------+
```

**Scala**
```scala
import org.apache.spark.sql.functions._
val df = Seq( (1, Seq(Map("k"->1), Map("k"->2))) ).toDF("id","items")
val exploded = df.withColumn("item", explode($"items")).select($"id", $"item".getItem("k").as("k"))
exploded.show(false)
```

**Python**
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

---

## 9) UDFs
**What it does:** Normalize strings to trimmed lowercase.

**Input**
```sql
+---+------+
|id |name  |
+---+------+
|1  | "  Amy " |
|2  | null |
+---+------+
```

**Scala**
```scala
import org.apache.spark.sql.functions.udf
val norm = udf((s: String) => if (s==null) "" else s.trim.toLowerCase)
val df = Seq((1,"  Amy "), (2,null.asInstanceOf[String])).toDF("id","name")
val out = df.withColumn("norm_name", norm($"name"))
out.show(false)
```

**Python**
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
```sql
+---+------+---------+
|id |name  |norm_name|
+---+------+---------+
|1  |  Amy |amy      |
|2  |null  |         |
+---+------+---------+
```
**Tip:** prefer built-ins / pandas_udf for performance.

---

## 10) Broadcast & Accumulators (broadcast filter)
**What it does:** Shares a small set of allowed countries across executors.

**Input**
```
+---+-------+
|id |country|
+---+-------+
|1  |US     |
|2  |MX     |
|3  |CA     |
+---+-------+
```

**Scala**
```scala
import org.apache.spark.sql.functions._
val df = Seq((1,"US"),(2,"MX"),(3,"CA")).toDF("id","country")
val br = spark.sparkContext.broadcast(Set("US","CA"))
val out = df.filter($"country".isin(br.value.toSeq:_*))
out.show(false)
```

**Python**
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

---

## 11) RDD Ops from DataFrame
**What it does:** Hop to RDD to do low-level pair counting.

**Input**
```
+---+----+
|id |val |
+---+----+
|1  |a   |
|1  |b   |
|2  |a   |
+---+----+
```

**Scala**
```scala
val df = Seq((1,"a"),(1,"b"),(2,"a")).toDF("id","val")
val r2 = df.rdd.map(r => (r.getAs[Int]("id"), 1)).reduceByKey(_ + _)
r2.collect().foreach(println)
```

**Python**
```python
df = spark.createDataFrame([(1,"a"),(1,"b"),(2,"a")], ["id","val"])
r2 = (df.rdd.map(lambda r: (r["id"], 1)).reduceByKey(lambda a,b: a+b))
print(sorted(r2.collect()))
```

**Output**
```
(1,2)
(2,1)
```

---

## 12) SQL Registration
**What it does:** Query a DF via SQL.

**Input**
```sql
+--------+-----+
|deviceId|temp |
+--------+-----+
|d1      |21.5 |
|d1      |22.0 |
|d2      |19.0 |
+--------+-----+
```

**Scala**
```scala
val df = Seq(("d1",21.5),("d1",22.0),("d2",19.0)).toDF("deviceId","temp")
df.createOrReplaceTempView("events")
val res = spark.sql("SELECT deviceId, COUNT(*) AS cnt FROM events GROUP BY deviceId")
res.orderBy("deviceId").show(false)
```

**Python**
```python
df = spark.createDataFrame([("d1",21.5),("d1",22.0),("d2",19.0)], ["deviceId","temp"])
df.createOrReplaceTempView("events")
res = spark.sql("SELECT deviceId, COUNT(*) AS cnt FROM events GROUP BY deviceId")
res.orderBy("deviceId").show(truncate=False)
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

---

## 13) Structured Streaming: readStream (Kafka) + parse JSON
**What it does:** Reads Kafka bytes, casts to string, parses JSON to columns.

**Input (Kafka messages value):**
```json
{"deviceId":"d1","ts":"2025-08-27T10:00:00Z","latency_ms":100}
{"deviceId":"d2","ts":"2025-08-27T10:01:00Z","latency_ms":200}
```

**Scala**
```scala
import org.apache.spark.sql.functions._, org.apache.spark.sql.types._
val schema = new StructType()
  .add("deviceId", "string").add("ts", "string").add("latency_ms","integer")

val src = spark.readStream.format("kafka")
  .option("kafka.bootstrap.servers","broker:9092")
  .option("subscribe","topic-in")
  .option("startingOffsets","earliest").load()

val parsed = src.selectExpr("CAST(value AS STRING) AS json")
  .select(from_json($"json", schema).as("data")).select("data.*")
```

**Python**
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

**Output (Console sink preview)**
```
+--------+-------------------------+----------+
|deviceId|ts                       |latency_ms|
+--------+-------------------------+----------+
|d1      |2025-08-27T10:00:00Z     |100       |
|d2      |2025-08-27T10:01:00Z     |200       |
+--------+-------------------------+----------+
```

---

## 14) Watermark + Window Aggregation (Event Time)
**What it does:** Event-time windowed avg with watermark for late data.

**Input:** (stream, same parsed as above).

**Scala**
```scala
import org.apache.spark.sql.functions._
val agg = parsed
  .withColumn("eventTime", $"ts".cast("timestamp"))
  .withWatermark("eventTime", "10 minutes")
  .groupBy(window($"eventTime","5 minutes"), $"deviceId")
  .agg(avg($"latency_ms").as("avg_latency"))
```

**Python**
```python
from pyspark.sql.functions import col, avg, window
agg = (parsed
       .withColumn("eventTime", col("ts").cast("timestamp"))
       .withWatermark("eventTime","10 minutes")
       .groupBy(window(col("eventTime"),"5 minutes"), col("deviceId"))
       .agg(avg(col("latency_ms")).alias("avg_latency")))
```

**Output (Console sink snapshot)**
```sql
+--------------------------+--------+-----------+
|window                    |deviceId|avg_latency|
+--------------------------+--------+-----------+
|[2025-08-27 10:00,10:05]  |d1      |100.0      |
|[2025-08-27 10:00,10:05]  |d2      |200.0      |
+--------------------------+--------+-----------+
```

---

## 15) Streaming Sinks & Triggers
**What it does:** Writes streaming results to console on a schedule.

**Input:** (any streaming DF, e.g., `agg` above)

**Scala**
```scala
val q = agg.writeStream.format("console")
  .option("checkpointLocation","/chk/agg")
  .outputMode("update")
  .trigger(processingTime="30 seconds")
  .start()
q.processAllAvailable() // for tests
```

**Python**
```python
q = (agg.writeStream.format("console")
     .option("checkpointLocation","/chk/agg")
     .outputMode("update")
     .trigger(processingTime="30 seconds")
     .start())
q.processAllAvailable()  # for tests
```

**Output (periodic batches)**
```sql
-------------------------------------------
Batch: 0
+--------------------------+--------+-----------+
|window                    |deviceId|avg_latency|
+--------------------------+--------+-----------+
|[2025-08-27 10:00,10:05]  |d1      |100.0      |
... (updates each trigger)
```

---

## 16) foreachBatch (JDBC Writes)
**What it does:** Per micro-batch, write to JDBC (upserts/inserts).

**Input:** (streaming `agg` DF)

**Scala**
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

**Python**
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

**Output:** Rows in the target table (e.g., `metrics`) after each batch.

---

## 17) Deduplication with Watermark
**What it does:** Drop duplicate events within watermark horizon.

**Input (stream):** Events with duplicates:
```
(d1, 2025-08-27T10:00:00Z, e1)
(d1, 2025-08-27T10:00:05Z, e1)  <-- duplicate event_id
```

**Scala**
```scala
val deduped = parsed
  .withColumn("eventTime", $"ts".cast("timestamp"))
  .withWatermark("eventTime", "10 minutes")
  .dropDuplicates("event_id","deviceId")
```

**Python**
```python
deduped = (parsed.withColumn("eventTime", col("ts").cast("timestamp"))
                 .withWatermark("eventTime","10 minutes")
                 .dropDuplicates(["event_id","deviceId"]))
```

**Output (Console)**
```sql
Only the first (earliest) (deviceId,event_id) row appears; later dups dropped.
```

---

## 18) mapGroupsWithState / flatMapGroupsWithState (Stateful)
**What it does:** Maintain per-key state across time (e.g., running average).

**Input (stream):**
```
(d1, t1, value=10), (d1, t2, value=14)
```

**Scala (sketch)**
```scala
import org.apache.spark.sql.streaming.{GroupState, GroupStateTimeout}
case class E(deviceId: String, ts: java.sql.Timestamp, value: Double)
case class Out(deviceId: String, avg: Double)
val ds = parsed.select($"deviceId",$"ts".cast("timestamp").as("ts"),$"value").as[E]
val out = ds.groupByKey(_.deviceId)
  .mapGroupsWithState(GroupStateTimeout.NoTimeout()) { (k, it, state) =>
    val (sum,cnt) = it.foldLeft((state.getOption.map(_._1).getOrElse(0.0),
                                  state.getOption.map(_._2).getOrElse(0))) {
      case ((s,c), e) => (s+e.value, c+1)
    }
    val avg = sum/cnt
    state.update((sum,cnt))
    Out(k, avg)
  }
```

**Python (sketch)**
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

**Output (Console)**
```
(d1, 12.0)   # running average after two rows
```

---

## 19) Datasets (Scala) vs DataFrames (Python)
**What it does:** Highlights type safety vs dynamic schema.

**Input:** (conceptual; same tables as prior examples)

**Scala**
```scala
case class User(id:Int, name:String)
val ds = Seq(User(1,"Amy")).toDS()  // typed Dataset[User]
ds.map(u => u.name.toUpperCase).show()
```

**Python**
```python
df = spark.createDataFrame([(1,"Amy")], ["id","name"])  # untyped DataFrame
df.selectExpr("upper(name) as NAME").show()
```

**Output**
**Scala:**
```sql
+---+
|AMY|
+---+
```
**Python:**
```sql
+----+
|NAME|
+----+
|AMY |
+----+
```

---

## 20) Porting Gotchas (illustrative)
**What it does:** Shows common pitfalls when translating Scala→Python.

**Input:** (various small DFs)
- Prefer built-ins over UDFs: `lower(trim(col("x")))` instead of Python UDF.
- Avoid `collect()` on big data: use `write`, `show`, or aggregations.
- Match streaming modes/watermarks: `update` vs `append` can change results.
- Define schemas (no case classes in Python).

**Scala → Python Example**
**Scala (good): built-ins**
```scala
df.withColumn("clean", lower(trim($"raw")))
```
**Python (good)**
```python
from pyspark.sql.functions import lower, trim, col
df.withColumn("clean", lower(trim(col("raw"))))
```

**Output:** Cleaned text with no UDF performance penalty.

---

## Appendix: Minimal creators used above
**Scala**
```scala
import spark.implicits._
```
**Python**
```python
# Already shown in sections; use spark.createDataFrame([...], ["col1","col2"])
```
