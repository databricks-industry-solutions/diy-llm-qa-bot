# Databricks notebook source
# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

import mlflow

# COMMAND ----------

dependencies = mlflow.pyfunc.get_model_dependencies(f"models:/{config['registered_model_name']}/Production")

# COMMAND ----------

# MAGIC %pip install -r {dependencies}

# COMMAND ----------

dbutils.library.restartPython()
import mlflow

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

eval_df = (
  spark
    .read
    .option("multiLine", "true")
    .json(
      f"{config['kb_documents_path']}/eval_data.json"
      )
    .toPandas()
  )

eval_df[:5]

# COMMAND ----------

# retrieve model from mlflow
model = mlflow.pyfunc.load_model(f"models:/{config['registered_model_name']}/Production")

# get a response
resp = model.predict(eval_df)

# COMMAND ----------

import pandas as pd

# COMMAND ----------

eval_df['retrieved_source']= pd.DataFrame(resp)['source']
eval_df['answer']= pd.DataFrame(resp)['answer']

# COMMAND ----------

spark.createDataFrame(eval_df).filter("correct_source=retrieved_source").count() / len(eval_df)

# COMMAND ----------

spark.createDataFrame(eval_df).display()

# COMMAND ----------


