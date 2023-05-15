# Databricks notebook source
# MAGIC %md ##Configuration
# MAGIC
# MAGIC In this notebook, we will capture all the configuration settings that support the work across those notebooks.

# COMMAND ----------

if 'config' not in locals():
  config = {}

# COMMAND ----------

config['kb_documents_path'] = "s3://db-gtm-industry-solutions/data/rcg/diy_llm_qa_bot/"
config['vector_store_path'] = '/dbfs/tmp/qabot/vector_store' # /dbfs/... is a local file system representation

# COMMAND ----------

config['database_name'] = 'qabot'

# create database if not exists
_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("solution-accelerator-cicd", "openai_api")

# COMMAND ----------

config['registered_model_name'] = 'databricks_llm_qabot_solution_accelerator'

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
model_name = "qabot"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/qabot'.format(username))

# COMMAND ----------


