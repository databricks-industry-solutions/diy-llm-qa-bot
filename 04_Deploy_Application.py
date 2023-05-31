# Databricks notebook source
# MAGIC %md The purpose of this notebook is to deploy the model to be used by the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will deploy the custom model registered with MLflow in the prior notebook and deploy it to Databricks model serving ([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)).  Databricks model serving provides containerized deployment options for registered models thought which authenticated applications can interact with the model via a REST API.  This provides MLOps teams an easy way to deploy, manage and integrate their models with various applications.

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# DBTITLE 1,Imports
import mlflow
import requests
import json
import time
from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

# DBTITLE 1,Retrieve the latest Production model version for deployment
latest_version = mlflow.MlflowClient().get_latest_versions(config['registered_model_name'], stages=['Production'])[0].version

# COMMAND ----------

# MAGIC %md ##Step 1: Deploy Model Serving Endpoint
# MAGIC
# MAGIC Models may typically be deployed to model serving endpoints using either the Databricks workspace user-interface or a REST API.  Because our model depends on the deployment of a sensitive environment variable, we will need to leverage a relatively new model serving feature that is currently only available via the REST API.
# MAGIC
# MAGIC See our served model config below and notice the `env_vars` part of the served model config - you can now store a key in a secret scope and pass it to the model serving endpoint as an environment variable.

# COMMAND ----------


served_models = [
    {
      "name": "current",
      "model_name": config['registered_model_name'],
      "model_version": latest_version,
      "workload_size": "Small",
      "scale_to_zero_enabled": "true",
      "env_vars": [{
        "env_var_name": "OPENAI_API_KEY",
        "secret_scope": config['openai_key_secret_scope'],
        "secret_key": config['openai_key_secret_key'],
      }]
    }
]
traffic_config = {"routes": [{"served_model_name": "current", "traffic_percentage": "100"}]}

# COMMAND ----------

# DBTITLE 1,Define functions to create or update endpoint according to our specification
def endpoint_exists():
  """Check if an endpoint with the serving_endpoint_name exists"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

def wait_for_endpoint():
  """Wait until deployment is ready, then return endpoint config"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

def create_endpoint():
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": config['serving_endpoint_name'], "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")
  
def update_endpoint():
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")

# COMMAND ----------

# DBTITLE 1,Use the defined function to create or update the endpoint
# gather other inputs the API needs
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

# kick off endpoint creation/update
if not endpoint_exists():
  create_endpoint()
else:
  update_endpoint()

# COMMAND ----------

# MAGIC %md You can use the link above to access the model serving endpoint we just created. 
# MAGIC
# MAGIC <img src='https://github.com/databricks-industry-solutions/diy-llm-qa-bot/raw/main/image/model_serving_ui.png'>

# COMMAND ----------

# MAGIC %md ##Step 2: Test Endpoint API

# COMMAND ----------

# MAGIC %md Next, we can use the code below to setup a function to query this endpoint.  This code is a slightly modified version of the code accessible through the *Query Endpoint* UI accessible through the serving endpoint page:

# COMMAND ----------

# DBTITLE 1,Define Functions to Query the Endpoint
import os
import requests
import numpy as np
import pandas as pd
import json

endpoint_url = f"""https://{serving_host}/serving-endpoints/{config['serving_endpoint_name']}/invocations"""


def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(dataset):
    url = endpoint_url
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()

# COMMAND ----------

# MAGIC %md And now we can test the endpoint:

# COMMAND ----------

# DBTITLE 1,Test the Model Serving Endpoint
# assemble question input
queries = pd.DataFrame({'question':[
  "What's the QPS limit for a serverless model serving request?"
]})

score_model( 
   queries
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC Some observed limitations:
# MAGIC * If we allow the endpoint to scale to zero, we will save cost when the bot is not queried. However, the first request after a long pause can take a few minutes, as it will require the endpoint to scale up from zero nodes
# MAGIC * The timeout limit for a serverless model serving request is 60 seconds. If more than 3 questions are submitted in the same request, the model may time out. 

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
