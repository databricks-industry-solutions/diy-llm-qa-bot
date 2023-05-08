# Databricks notebook source
# DBTITLE 1,Kafka config - see the RUNME notebook for instructions on setting up secrets
kafka_bootstrap_servers = dbutils.secrets.get("solution-accelerator-cicd", "iot-anomaly-kafka-bootstrap-server")
security_protocol = "SASL_SSL"
sasl_mechanism = "PLAIN"
sasl_username = dbutils.secrets.get("solution-accelerator-cicd", "iot-anomaly-sasl-username")
sasl_password = dbutils.secrets.get("solution-accelerator-cicd", "iot-anomaly-sasl-password")
topic = "iot_msg_topic"
sasl_config = f'org.apache.kafka.common.security.plain.PlainLoginModule required username="{sasl_username}" password="{sasl_password}";'

# COMMAND ----------

# DBTITLE 1,Streaming checkpoint location
checkpoint_path = "/dbfs/tmp/iot-anomaly-detection/checkpoints"

# COMMAND ----------

# DBTITLE 1,Database settings
database = "rvp_iot_sa"

spark.sql(f"create database if not exists {database}")

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
model_name = "iot_anomaly_detection_xgboost"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/iot_anomaly_detection'.format(username))

# COMMAND ----------


