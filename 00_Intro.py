# Databricks notebook source
# MAGIC %md The purpose of this notebook is to set the various configuration values that will control the notebooks that make up the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC The goal of this solution accelerator is to show how we can leverage a large language model in combination with our own data to create an interactive application capable of answering questions specific to a particular domain or subject area.  The core pattern behind this is the delivery of a question along with a document or document fragment that provides relevant context for answering that question to the model.  The model will then respond with an answer that takes into consideration both the question and the context.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_flow.png' width=500>
# MAGIC
# MAGIC </p>
# MAGIC To assemble this application, *i.e.* the Q&A Bot, we will need to assemble a series of documents that are relevant to the domain we wish to serve.  We will need to index these to enable rapid search given a user question. We will then need to assemble the core application which combines a question with a document to form a prompt and submits that prompt to a model in order to generate a response. Finally, we'll need to package both the indexed documents and the core application component as a microservice to enable a wide range of deployment options.
# MAGIC
# MAGIC We will tackle these three steps across the following three notebooks:</p>
# MAGIC
# MAGIC * 01: Build Document Index
# MAGIC * 02: Assemble Application
# MAGIC * 03: Deploy Application
# MAGIC </p>

# COMMAND ----------

# MAGIC %md Initialize the paths we will use throughout the accelerator

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

dbutils.fs.rm(config['vector_store_path'][5:], True)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
