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

# MAGIC %md ##Step 1: Deploy Model Serving Endpoint
# MAGIC
# MAGIC Models may typically be deployed to model sharing endpoints using either the Databricks workspace user-interface or a REST API.  Because our model depends on the deployment of a sensitive environment variable, we will need to leverage a relatively new model serving feature that is currently only available via the REST API.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Step 2: Test Endpoint API

# COMMAND ----------



# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
