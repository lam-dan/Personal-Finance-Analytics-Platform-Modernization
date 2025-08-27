# RDD Ops: Inputs, Outputs, and What They Do

## 1) parallelize, map, filter

### What it does:
Create an RDD in memory, transform each element (map), and keep only those matching a predicate (filter).
Transformations are lazy; nothing runs until an action (e.g., collect) is called.

### Input
```
[1, 2, 3, 4]
```

### Scala
```scala
val rdd      = sc.parallelize(Seq(1, 2, 3, 4))
val doubled  = rdd.map(x => x * 2)        // [2, 4, 6, 8]
val evens    = doubled.filter(_ % 2 == 0) // [2, 4, 6, 8] (all even)
val out      = evens.collect()
```

### Python
```python
rdd     = sc.parallelize([1, 2, 3, 4])
doubled = rdd.map(lambda x: x * 2)        // [2, 4, 6, 8]
evens   = doubled.filter(lambda x: x % 2 == 0)
out     = evens.collect()
```

### Output
```
[2, 4, 6, 8]
```

## 2) Key-Value RDDs (map to pairs, reduceByKey)

### What it does:
Turn items into (key, value) pairs, then combine values by key in a distributed way. `reduceByKey` is pre-aggregating on partitions (more efficient than `groupByKey` for numeric combos).

### Input
```
[1, 2, 2, 3]
```

### Scala
```scala
val rdd    = sc.parallelize(Seq(1, 2, 2, 3))
val pairs  = rdd.map(x => (x, 1))              // [(1,1),(2,1),(2,1),(3,1)]
val counts = pairs.reduceByKey(_ + _)          // [(1,1),(2,2),(3,1)]
val out    = counts.collect().toList
```

### Python
```python
rdd     = sc.parallelize([1, 2, 2, 3])
pairs   = rdd.map(lambda x: (x, 1))            // [(1,1),(2,1),(2,1),(3,1)]
counts  = pairs.reduceByKey(lambda a,b: a+b)   // [(1,1),(2,2),(3,1)]
out     = counts.collect()
```

### Output
```
[(1, 1), (2, 2), (3, 1)]
```

## 3) flatMap (tokenize lines)

### What it does:
Map each input to 0..n outputs and then flatten. Common for splitting text into words.

### Input
```
["hello world", "hello spark"]
```

### Scala
```scala
val lines = sc.parallelize(Seq("hello world", "hello spark"))
val words = lines.flatMap(_.split(" "))
val out   = words.collect().toList
```

### Python
```python
lines = sc.parallelize(["hello world", "hello spark"])
words = lines.flatMap(lambda s: s.split(" "))
out   = words.collect()
```

### Output
```
["hello", "world", "hello", "spark"]
```

## 4) groupByKey vs reduceByKey

### What it does:
`groupByKey` collects all values per key into an iterable (can be heavy).
`reduceByKey` aggregates during shuffle (usually preferred for sums/aggregates).

### Input
Pairs: `[("a", 1), ("a", 2), ("b", 5)]`

### Scala
```scala
val pairs   = sc.parallelize(Seq(("a",1),("a",2),("b",5)))
val grouped = pairs.groupByKey().mapValues(_.toList)  // ("a",[1,2]), ("b",[5])
val reduced = pairs.reduceByKey(_ + _)                // ("a",3), ("b",5)
(grouped.collect().toList, reduced.collect().toList)
```

### Python
```python
pairs   = sc.parallelize([("a",1),("a",2),("b",5)])
grouped = pairs.groupByKey().mapValues(list)          // ("a",[1,2]), ("b",[5])
reduced = pairs.reduceByKey(lambda a,b: a+b)          // ("a",3), ("b",5)
(grouped.collect(), reduced.collect())
```

### Output
**grouped:**
```
[("a", [1, 2]), ("b", [5])]
```
**reduced:**
```
[("a", 3), ("b", 5)]
```

## 5) Common Actions (collect, count, first, take)

### What it does:
Actions trigger computation and return results to the driver (be careful with size).

### Input
```
[10, 20, 30, 40]
```

### Scala
```scala
val rdd = sc.parallelize(Seq(10, 20, 30, 40))
val all  = rdd.collect().toList // [10,20,30,40]
val n    = rdd.count()          // 4
val fst  = rdd.first()          // 10
val t2   = rdd.take(2).toList   // [10,20]
```

### Python
```python
rdd  = sc.parallelize([10, 20, 30, 40])
all_ = rdd.collect()   # [10,20,30,40]
n    = rdd.count()     # 4
fst  = rdd.first()     # 10
t2   = rdd.take(2)     # [10,20]
```

### Output
```
collect → [10, 20, 30, 40]
count → 4
first → 10
take(2) → [10, 20]
```

## 6) Pair RDD Joins

### What it does:
Relational join on keys for (K, V) and (K, W) RDDs. Produces (K, (V, W)).

### Input
```
rdd1 = [(1,"a"), (2,"b"), (3,"c")]
rdd2 = [(1,"x"), (2,"y")]
```

### Scala
```scala
val rdd1  = sc.parallelize(Seq((1,"a"), (2,"b"), (3,"c")))
val rdd2  = sc.parallelize(Seq((1,"x"), (2,"y")))
val join  = rdd1.join(rdd2)
val left  = rdd1.leftOuterJoin(rdd2)
(join.collect().toList, left.collect().toList)
```

### Python
```python
rdd1 = sc.parallelize([(1,"a"), (2,"b"), (3,"c")])
rdd2 = sc.parallelize([(1,"x"), (2,"y")])
join = rdd1.join(rdd2)              # inner
left = rdd1.leftOuterJoin(rdd2)     # left outer
(join.collect(), left.collect())
```

### Output
**join:**
```
[(1, ("a", "x")), (2, ("b", "y"))]
```
**leftOuterJoin:**
```
[(1, ("a", "x")), (2, ("b", "y")), (3, ("c", None))]
```

## 7) Save & Load (saveAsTextFile, textFile)

### What it does:
Persist an RDD as a directory of part files; read text files into an RDD of strings.

### Input
RDD: `["one", "two", "three"]`

### Scala
```scala
val rdd = sc.parallelize(Seq("one","two","three"))
rdd.saveAsTextFile("s3://bucket/out")   // writes part-00000...
val loaded = sc.textFile("s3://bucket/out")
val out    = loaded.collect().toList    // ["one","two","three"]
```

### Python
```python
rdd = sc.parallelize(["one","two","three"])
rdd.saveAsTextFile("s3://bucket/out")
loaded = sc.textFile("s3://bucket/out")
out = loaded.collect()
```

### Output
```
["one", "two", "three"]
```

## Extra: Classic Word Count (flatMap → map → reduceByKey)

### What it does:
Tokenize, pair each word with 1, then sum counts per word.

### Input
```
["to be or not", "to be"]
```

### Scala
```scala
val lines  = sc.parallelize(Seq("to be or not", "to be"))
val counts = lines
  .flatMap(_.split("\\s+"))
  .map(w => (w, 1))
  .reduceByKey(_ + _)
  .collect()
  .toList
```

### Python
```python
lines  = sc.parallelize(["to be or not", "to be"])
counts = (lines
          .flatMap(lambda s: s.split())
          .map(lambda w: (w, 1))
          .reduceByKey(lambda a,b: a+b)
          .collect())
```

### Output (order may vary)
```
[("to", 2), ("be", 2), ("or", 1), ("not", 1)]
```

## Mental model: Transformations vs. Actions

**Transformations** (e.g., `map`, `filter`, `flatMap`, `reduceByKey`, `groupByKey`, `join`) are lazy and build a lineage (DAG).

**Actions** (e.g., `collect`, `count`, `take`, `first`, `saveAsTextFile`) execute the DAG and return/commit results.