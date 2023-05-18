# Databricks notebook source
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Set document path
config['kb_documents_path'] = "s3://db-gtm-industry-solutions/data/rcg/diy_llm_qa_bot/"
config['vector_store_path'] = '/dbfs/tmp/qabot/vector_store' # /dbfs/... is a local file system representation

# COMMAND ----------

# DBTITLE 1,Create database
config['database_name'] = 'qabot'

# create database if not exists
_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Set Environmental Variables for tokens
import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("solution-accelerator-cicd", "openai_api")

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
config['registered_model_name'] = 'databricks_llm_qabot_solution_accelerator'
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

config["eval_dataset_path"]= "./data/eval_data.tsv"
config['openai_key_secret_scope'] = "solution-accelerator-cicd" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope name you actually use 
config['openai_key_secret_key'] = "openai_api" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope key name you actually use
config['serving_endpoint_name'] = "qa-bot-endpoint"

# COMMAND ----------


