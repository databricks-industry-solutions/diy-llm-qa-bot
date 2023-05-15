# Databricks notebook source
# MAGIC %md The purpose of this notebook is to access and prepare our data for use with the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC So that our qabot application can respond to user questions with relevant answers, we will provide our model with content from documents relevant to the question being asked.  The idea is that the bot will leverage the information in these documents as it formulates a response.
# MAGIC
# MAGIC For our application, we've extracted a series of documents from the [Databricks Knowledge Base](https://kb.databricks.com/).  This is an online forum where frequently asked questions are addressed with high-quality, detailed responses.  Using these documents to provide context will allow our bot to respond to questions relevant to this subject area with deep expertise.
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
# MAGIC %pip install langchain==0.0.166 tiktoken==0.4.0 openai==0.27.6 faiss-cpu==1.7.4

# COMMAND ----------

# DBTITLE 1,Import Required Functions
import pyspark.sql.functions as fn

import json

from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: Load the Raw Data to Table
# MAGIC
# MAGIC Our first step is to access the extracted Knowledge Base documents. We can load them to a table using a Spark DataReader configured for reading [JSON](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrameReader.json.html).  This reader will read the first few JSON documents in our file to infer a schema for out dataset, greatly simplifying the process of accessing the data:

# COMMAND ----------

# DBTITLE 1,Read JSON Data to Dataframe
raw = (
  spark
    .read
    .json(
      f"{config['kb_documents_path']}/raw-documents.json"
      )
    .withColumn('created_at', fn.to_timestamp( 'created_at', "yyyy-MM-dd'T'HH:mm:ss[.SSS][XXX]"))    
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
    .saveAsTable('answers')
  )

# count rows in table
print(spark.table('answers').count())

# COMMAND ----------

# MAGIC %md ##Step 2: Prepare Data for Indexing
# MAGIC
# MAGIC While there are many fields avaiable to us in our newly loaded table, the fields that are relevant for our application are:
# MAGIC </p>
# MAGIC
# MAGIC * name - the primary title for the knowledge base page
# MAGIC * description - the descriptive subtitle for the knowledge base page
# MAGIC * answer - the detailed response which may include more information about the problem
# MAGIC * url - the url pointing to the online version of the knowledge base page
# MAGIC </p>
# MAGIC
# MAGIC The answer field is a structure which includes body, body_txt, format and other elements. In this dataset, all the answers are formatted as html. The raw html is accessible through the body attribute while the text displayed on the resulting html page is accessible through the body_txt field. We'll grab our answer data from this latter field.
# MAGIC
# MAGIC We've also identified that there are a few bad records in our dataset as indicated by short or missing url, name or answer attributes. We'll exclude these from our work:

# COMMAND ----------

# DBTITLE 1,Retrieve Raw Inputs
raw_inputs = (
  spark
    .table('answers')
    .selectExpr(
      'name',
      'description',
      'answer.body_txt as answer',
      'url'
      )
    .filter('len(url) > 0') # remove records with bad urls
    .filter('len(answer) >= 2') # remove records with bad answers
    .filter('len(name) >= 5') # remove records with bad names
  ) 

display(raw_inputs)

# COMMAND ----------

# MAGIC %md The content available within each doc varies but some documents can be quite long.  Here is an example of a large document in our dataset:

# COMMAND ----------

# DBTITLE 1,Retrieve a Large Answer
answer = (
  raw_inputs
    .select('answer') # get just the answer field
    .orderBy(fn.expr("len(answer)"), ascending=False) # sort by length
    .limit(1) # get top 1
     .collect()[0]['answer'] # pull answer to a variable
  )

# display answer
print(answer)

# COMMAND ----------

# MAGIC %md The process of converting a document to an index involves us translating it to a fixed-size embedding.  An embedding is a set of numerical values, kind of like a coordinate, that summarizes the content in a unit of text. While large embeddings are capable of capturing quite a bit of detail about a document, the larger the document submitted to it, the more the embedding generalizes the content.  It's kind of like asking someone to summarize a paragraph, a chapter or an entire book within a fixed number of words.  The greater the scope, the more the summary must eliminate detail and focus on the higher-level concepts in the text.
# MAGIC
# MAGIC A common strategy for dealing with this when generating embeddings is to divide the text into chunks.  These chunks need to be large enough to capture meaningful detail but not so large that key elements get washed out in the generalization.  Its more of an art than a science to determine an appropriate chunk size, but here we'll use a very small chunk size to illustrate what's happening in this step:

# COMMAND ----------

# DBTITLE 1,Split Answer into Chunks
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
for chunk in text_splitter.split_text(answer):
  print(chunk, '\n')

# COMMAND ----------

# MAGIC %md Please note that we are specifying overlap between our chunks.  This is to help avoid the arbitrary separation of words that might capture a key concept. 
# MAGIC
# MAGIC We have set our overlap size very small for this demonstration but you may notice that overlap size does not neatly translate into the exact number of words that will overlap between chunks. This is because we are not splitting the content directly on words but instead on byte-pair encoding tokens derived from the words that make up the text.  You can learn more about byte-pair encoding [here](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) but just note that its a frequently employed mechanism for compressing text in many LLM algorithms.

# COMMAND ----------

# MAGIC %md With the concept of document splitting under our belt, let's write a function to divide our documents into chunks and apply it to our data:

# COMMAND ----------

# DBTITLE 1,Chunking Configurations
chunk_size = 2500
chunk_overlap = 400

# COMMAND ----------

# DBTITLE 1,Divide Inputs into Chunks
@fn.udf('array<string>')
def get_chunks(answer):

  # instantiate tokenization utilities
  text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  
  # split answer into chunks
  return text_splitter.split_text(answer)


# split answer into chunks
chunked_inputs = (
  raw_inputs
    .withColumn('chunks', get_chunks('answer')) # divide answers into chunks
    .drop('answer')
    .withColumn('num_chunks', fn.expr("size(chunks)"))
    .withColumn('chunk', fn.expr("explode(chunks)"))
    .drop('chunks')
    .withColumnRenamed('chunk','answer')
  )

  # display transformed data
display(chunked_inputs)

# COMMAND ----------

# MAGIC %md ##Step 4: Create Vector Store
# MAGIC
# MAGIC With our data divided into chunks, we are ready to convert these records into searchable embeddings. Our first step is to separate the content that will be converted from the content that will serve as the metadata surrounding the document:

# COMMAND ----------

# DBTITLE 1,Separate Inputs into Searchable Text & Metadata
# convert inputs to pandas dataframe
inputs = chunked_inputs.toPandas()

# extract searchable text elements
text_inputs = inputs['answer'].to_list()

# extract metadata
metadata_inputs = (
  inputs
    .drop(['answer','num_chunks'], axis=1)
    .to_dict(orient='records')
  )

# COMMAND ----------

# MAGIC %md Next, we will initialize the vector store into which we will load our data.  If you are not familiar with vector stores, these are specialized databases that store text data as embeddings and enable fast searches based on content similarity.  We will be using the [FAISS vector store](https://faiss.ai/) developed by Facebook AI Research. It's fast and lightweight, characteristics that make it ideal for our scenario.
# MAGIC
# MAGIC The key to setting up the vector store is to configure it with an embedding model that it will used to convert both the documents and any searchable text to an embedding (vector). You have a wide range of choices avaialble to you as you consider which embedding model to employ.  Some popular models include the [sentence-transformer](https://huggingface.co/models?library=sentence-transformers&sort=downloads) family of models available on the HuggingFace hub as well as the [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings):
# MAGIC
# MAGIC **NOTE** The OpenAI API Key required with the OpenAIEmbeddings object is being set in notebook 00 and is made available as an environment variable per [these instructions](https://python.langchain.com/en/latest/ecosystem/openai.html#installation-and-setup).

# COMMAND ----------

# DBTITLE 1,Load Vector Store
# identify embedding model that will generate embedding vectors
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# instantiate vector store object
vector_store = FAISS.from_texts(
  embedding=embeddings, 
  texts=text_inputs, 
  metadatas=metadata_inputs
  )

# COMMAND ----------

# MAGIC %md So that we make use of our vector store in subsequent notebooks, let's persist it to storage:

# COMMAND ----------

# DBTITLE 1,Persist Vector Store to Storage
vector_store.save_local(folder_path=config['vector_store_path'])

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
