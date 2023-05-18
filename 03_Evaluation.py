# Databricks notebook source
# MAGIC %md The purpose of this notebook is to evaluate the model to be used by the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ## Evaluating QA model performance
# MAGIC In this notebook, we showcase how to evaluate QA performance using langchain's `QAEvalChain` using a evaluation set containing correct references and responses. We use an LLM as a grader to compare model responses with the correct responses. 

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Scoring the evaluation set
# MAGIC
# MAGIC We first retrieve and install the Python dependencies from the logged model in order to score the model. 
# MAGIC
# MAGIC mlflow writes a file with model dependencies at `requirements_path` in DBFS. We then use %pip to install the dependencies in the file.  

# COMMAND ----------

import mlflow

requirements_path = mlflow.pyfunc.get_model_dependencies(config['model_uri'])
%pip install -r $requirements_path
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Because the previous block restarts the kernel, let's set our configs again and import the dependencies:

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

import pandas as pd
import numpy as np
import openai
import json
from langchain.llms import OpenAI
import os
from langchain.evaluation.qa import QAEvalChain
import mlflow

# COMMAND ----------

# MAGIC %md We have prepared an evaluation set of questions and correct answers that contain both the correct reference and a sample answer. Let's take a look at an example: 

# COMMAND ----------

eval_dataset = pd.read_csv(config['eval_dataset_path'], sep='\t').to_dict('records')
eval_dataset[0] 

# COMMAND ----------

# DBTITLE 1,Score eval dataset with the logged model
queries = pd.DataFrame({'question': [r['question'] for r in eval_dataset]})
model = mlflow.pyfunc.load_model(config['model_uri'])
predictions = model.predict(queries)
predictions[0]

# COMMAND ----------

# MAGIC %md langchain's `QAEvalChain` acts as a grader: compares whether the scored answers are sufficiently similar to the ground truth and returns CORRECT or INCORRECT for each question.

# COMMAND ----------

llm = OpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(eval_dataset, predictions, question_key="question", prediction_key="answer")
graded_outputs[:5]

# COMMAND ----------

# MAGIC %md We can compile the graded results, the scored answers and the eval dataset back into one DataFrame. Note that the bot can sometimes produce correct answers based on different reference sources.

# COMMAND ----------

results = pd.DataFrame(
  [{
    "question": eval_data["question"], 
    "prediction": predict["answer"], 
    "source": predict["source"], 
    "correct_source": eval_data["correct_source"], 
    "answer": eval_data["answer"], 
    "find_correct_source": predict["source"] == eval_data["correct_source"], 
    "same_as_answer": True if graded_output['text'].strip() == 'CORRECT' else False
    } 
    for (predict, eval_data, graded_output) in zip(predictions, eval_dataset, graded_outputs)])
display(spark.createDataFrame(results))

# COMMAND ----------

# DBTITLE 1,How often does the model produce correct responses on this eval set according to our grader?
results['same_as_answer'].sum() / len(results)

# COMMAND ----------

# MAGIC %md 
# MAGIC Our QA bot seems to produce sensible responses most of the time according to the LLM grader. However, it is still important for the developer to regularly evaluate the performance by reading the responses. It is often possible for the LLM to miss nuanced differences between concepts and produce false negative gradings, especially when answers are long and complex.  And make sure to review the eval question set periodically so that it reflects the type of questions the users submit. 

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |

# COMMAND ----------


