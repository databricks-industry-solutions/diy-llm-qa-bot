# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## IoT Data Generation
# MAGIC 
# MAGIC <img src="https://github.com/databricks-industry-solutions/iot-anomaly-detection/raw/main/resource/images/02_generate_iot_data.jpg">
# MAGIC 
# MAGIC In this notebook, we use `dbldatagen` to generate fictitious data and push into a Kafka topic.

# COMMAND ----------

# MAGIC %md
# MAGIC Generate the Data

# COMMAND ----------

# MAGIC %run ./notebook-config

# COMMAND ----------

import dbldatagen as dg
import dbldatagen.distributions as dist
from pyspark.sql.types import IntegerType, FloatType, StringType, LongType

states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY' ]

table_name = "iot_stream_example_"
spark.sql(f"drop table if exists {table_name}")

data_rows = 2000
df_spec = (
  dg.DataGenerator(
    spark,
    name="test_data_set1",
    rows=data_rows,
    partitions=4
  )
  .withIdOutput()
  .withColumn("device_id", IntegerType(), minValue=1, maxValue=1000)
  .withColumn(
    "device_model",
    StringType(),
    values=['mx2000', 'xft-255', 'db-1000', 'db-2000', 'mlr-120'],
    random=True
  )
  .withColumn("timestamp", LongType(), minValue=1577833200, maxValue=1673714337, random=True)
  .withColumn("sensor_1", IntegerType(), minValue=-10, maxValue=100, random=True, distribution=dist.Gamma(40.0,9.0))
  .withColumn("sensor_2", IntegerType(), minValue=0, maxValue=10, random=True)
  .withColumn("sensor_3", FloatType(), minValue=0.0001, maxValue=1.0001, random=True)
  .withColumn("state", StringType(), values=states, random=True)
)
                            
df = df_spec.build()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Write to Kafka

# COMMAND ----------

from pyspark.sql.functions import to_json, struct, col, cast
from pyspark.sql.types import BinaryType
import time

#Get the data ready for Kafka
kafka_ready_df = (
                  df.select(
                    col("id").cast(BinaryType()).alias("key"),
                    to_json(
                      struct(
                        [col(column) for column in df.columns]
                      )
                    ).cast(BinaryType()).alias("value")
                  )
)

display(kafka_ready_df)

# COMMAND ----------

options = {
    "kafka.ssl.endpoint.identification.algorithm": "https",
    "kafka.sasl.jaas.config": sasl_config,
    "kafka.sasl.mechanism": sasl_mechanism,
    "kafka.security.protocol" : security_protocol,
    "kafka.bootstrap.servers": kafka_bootstrap_servers,
    "group.id": 1,
    "subscribe": topic,
    "topic": topic,
    "checkpointLocation": checkpoint_path
}

#Write to Kafka
(
  kafka_ready_df
    .write
    .format("kafka")
    .options(**options)
    .save()
)
