# Databricks notebook source
# MAGIC %md The purpose of this notebook is to fine tune the embedding models based on our data for use with the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC So that our qabot application can respond to user questions with relevant answers, we will provide our model with content from documents relevant to the question being asked.  The idea is that the bot will leverage the information in these documents as it formulates a response.
# MAGIC
# MAGIC For our application, we've extracted a series of documents from [Databricks documentation](https://docs.databricks.com/), [Spark documentation](https://spark.apache.org/docs/latest/), and the [Databricks Knowledge Base](https://kb.databricks.com/).  Databricks Knowledge Base is an online forum where frequently asked questions are addressed with high-quality, detailed responses.  Using these three documentation sources to provide context will allow our bot to respond to questions relevant to this subject area with deep expertise.
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_data_processing4.png' width=700>
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC In this notebook, we will load these documents, extracted as a series of JSON documents through a separate process, to a table in the Databricks environment.  We will retrieve those documents along with metadata about them and feed that to a vector store which will create on index enabling fast document search and retrieval.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install sentence-transformers==2.2.2

# COMMAND ----------

# DBTITLE 1,Import Required Functions
import pyspark.sql.functions as fn

import json

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: Load the Raw Data to Table
# MAGIC
# MAGIC A snapshot of the three documentation sources is made available at a publicly accessible cloud storage. Our first step is to access the extracted documents. We can load them to a table using a Spark DataReader configured for reading [JSON](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrameReader.json.html) with the `multiLine` option.  

# COMMAND ----------

# DBTITLE 1,Read JSON Data to Dataframe
raw = (
  spark
    .read
    .option("multiLine", "true")
    .json(
      f"{config['kb_documents_path']}/source/"
      )
  )

display(raw)

# COMMAND ----------

# MAGIC %md We can persist this data to a table as follows:

# COMMAND ----------

# DBTITLE 1,Save Data to Table
# save data to table
_ = (
  raw
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('sources')
  )

# count rows in table
print(spark.table('sources').count())

# COMMAND ----------

# MAGIC %md ##Step 2: Prepare Data for Fine Tuning Embeddings
# MAGIC
# MAGIC While there are many fields avaiable to us in our newly loaded table, the fields that are relevant for our application are:
# MAGIC
# MAGIC * text - Documentation text or knowledge base response which may include relevant information about user's question
# MAGIC * source - the url pointing to the online document

# COMMAND ----------

# DBTITLE 1,Retrieve Raw Inputs
raw_inputs = (
  spark
    .table('sources')
    .selectExpr(
      'text',
      'source'
      )
  ) 

display(raw_inputs)

# COMMAND ----------

import re

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, StructType, StructField

def markdown_to_plain_text(document):
    plain_text = re.sub(r'<[^>]*>', ' ', document)  # Remove HTML tags
    plain_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', plain_text)  # Replace links with link text
    plain_text = re.sub(r'\n\d+\.', ' ', plain_text)  # Replace numbered list with spaces
    plain_text = re.sub(r'#+', ' ', plain_text)  # Replace headers with spaces
    plain_text = re.sub(r'-{3,}', '', plain_text)  # Remove series of "-"
    plain_text = re.sub(r'(\n-|\n){2,}', ' ', plain_text)  # Remove extra newlines
    plain_text = re.sub(r'(\n\n){2,}', ' ', plain_text)  # Remove excessive newlines
    plain_text = re.sub(r'<!--.*?-->', '', plain_text, flags=re.DOTALL)  # Remove comments
    plain_text = re.sub(r'\n\.{2,}.*?(\n|$)', '', plain_text)  # Remove rst note blocks
    plain_text = re.sub(r'\n\*{3}.*?(\n|$)', '', plain_text)  # Remove horizontal rules
    plain_text = re.sub(r'\[\[(.*?)\]\]\([^)]+\)', r'\1', plain_text)  # Remove double brackets and associated URLs
    plain_text = re.sub(r'```.*?```', '', plain_text, flags=re.DOTALL)  # Remove code blocks
    plain_text = re.sub(r'\n', ' ', plain_text)  # Remove all newlines
    plain_text = re.sub(r'[\*_]', '', plain_text)  # Remove other markdown artifacts
    plain_text = re.sub(r'\s*\|\s*', ' ', plain_text)  # Remove "|" and ensure only one space between words
    plain_text = re.sub(r'!check marked yes', '', plain_text)  # Remove "!check marked yes" phrase
    plain_text = re.sub(r'\s{2,}', ' ', plain_text)  # Remove multiple spaces between words
    plain_text = re.sub(r'\s+\.\s+', '. ', plain_text)  # Fix spaces around periods
    plain_text = plain_text.strip()
    return plain_text


def extract_title_and_text(document):
  array_of_text = markdown_to_plain_text(document).split("=")
  title, text = array_of_text[0].strip(), array_of_text[-1].strip()
  return title, text

schema = StructType([  
    StructField("title", StringType(), False),  
    StructField("text", StringType(), False)  
])

markDownToPlainUDF = udf(extract_title_and_text, schema) 

# COMMAND ----------

clean_inputs = raw_inputs.withColumn('extracted', markDownToPlainUDF(col('text'))) \
  .select(col("extracted.*"), col("source"))
display(clean_inputs)

# COMMAND ----------

# MAGIC %md
# MAGIC Define the training dataset.

# COMMAND ----------

from sentence_transformers import InputExample

def create_dataset_for_multiple_loss(input_df):
  pandasDF = input_df.toPandas()
  train_examples = []
  for _, row in pandasDF.iterrows():
    query, text = row["title"], row["text"]
    train_examples.append(InputExample(texts=[query, text]))
  return train_examples

train_examples = create_dataset_for_multiple_loss(clean_inputs)

# COMMAND ----------

from sentence_transformers import InputExample, models, SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import losses
from accelerate import notebook_launcher

from sentence_transformers import models, SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import losses

model_name = "intfloat/e5-large-unsupervised"

word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4) # INFO: If you run into CUDA out of memory issues, reduce batch_size
# INFO: This is the triplet loss
train_loss = losses.MultipleNegativesRankingLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

# COMMAND ----------

config['embedding_model_path'] = "/dbfs/tmp/qabot/qabot-embedding"

# COMMAND ----------

model.save(config['embedding_model_path'])

# COMMAND ----------

# Save the model as Pyfunc to mlflow
import mlflow
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
 
class SentenceTransformerEmbeddingModel(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    self.model = SentenceTransformer(context.artifacts["sentence-transformer-model"], device=device)
    
  def predict(self, context, model_input): 
    texts = model_input.iloc[:,0].to_list() # get the first column
    sentence_embeddings = self.model.encode(texts)
    return pd.Series(sentence_embeddings.tolist())

# COMMAND ----------

import mlflow
from mlflow.utils.environment import _mlflow_conda_env
import accelerate
import sentence_transformers
import cloudpickle
EMBEDDING_CONDA_ENV = _mlflow_conda_env(
    additional_pip_deps=[
        f"accelerate=={accelerate.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
        f"sentence-transformers=={sentence_transformers.__version__}",
    ]
)

with mlflow.start_run() as run:
  embedding_model = SentenceTransformerEmbeddingModel()
  model_info = mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=embedding_model, 
    input_example=["spark overview"], 
    artifacts={"sentence-transformer-model": config['embedding_model_path']}, 
    conda_env=EMBEDDING_CONDA_ENV,
    registered_model_name=config['embedding_registered_model_name']
    )

# COMMAND ----------

import mlflow
run_id = run.info.run_id
logged_model_uri = f"runs:/{run_id}/model"

# logged_model_uri = 'runs:/1d8e4e885e36440cb1f1f16fa9415117/model'
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model_uri, result_type='string')

# COMMAND ----------

import pandas as pd

test_df = pd.DataFrame(['pyspark.pandas.MultiIndex.unique',
              'RobustScalarModel'], columns=["text"])
sample = spark.createDataFrame(test_df)
summaries = sample.select(sample.text, loaded_model(sample.text).alias("embeddings"))
display(summaries)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
