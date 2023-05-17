# Databricks notebook source
dbutils.widgets.text("model_uri", "models:/qa-bot/production", "Model URI")
dbutils.widgets.text("openai_key_secret_scope", "qa_bot", "The secret scope of OpenAI API Key")
dbutils.widgets.text("eval_dataset_path", "eval_data.json", "File path of evaluation dataset")

# COMMAND ----------

# We install the Python dependencies needed to score the model
import mlflow
model_uri = dbutils.widgets.get("model_uri")
requirements_path = model_uri = mlflow.pyfunc.get_model_dependencies(model_uri)
%pip install -r $requirements_path
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import struct
import openai
import json
from langchain.llms import OpenAI
import os

# COMMAND ----------

eval_dataset_path = dbutils.widgets.get("eval_dataset_path")
with open(eval_dataset_path) as f:
  eval_dataset = json.load(f)
eval_dataset = eval_dataset


# COMMAND ----------

model_uri = dbutils.widgets.get("model_uri")
openai_key_secret_scope = dbutils.widgets.get("openai_key_secret_scope")
openai_key_secret_key = "openai_api_key"
model = mlflow.pyfunc.load_model(model_uri)
openai.api_key = dbutils.secrets.get(scope=openai_key_secret_scope, key=openai_key_secret_key)
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope=openai_key_secret_scope, key=openai_key_secret_key)

# COMMAND ----------

questions = [data["question"] for data in eval_dataset]
predictions = model.predict(questions)

# COMMAND ----------

from langchain.evaluation.qa import QAEvalChain

llm = OpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(eval_dataset, predictions, question_key="question", prediction_key="answer")

# COMMAND ----------

results = [{"prediction": predict["answer"], "question": eval_data["question"], "source": predict["source"], "correct_source": eval_data["correct_source"], "find_correct_source": predict["source"] == eval_data["correct_source"], "answer": eval_data["answer"], "answer_source": eval_data["answer_source"], "same_as_answer": graded_output['text']} for (predict, eval_data, graded_output) in zip(predictions, eval_dataset, graded_outputs)]

# COMMAND ----------

results
