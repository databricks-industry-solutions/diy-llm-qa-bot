# Databricks notebook source
# MAGIC %md The purpose of this notebook is to define and persist the model to be used by the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our documents indexed, we can now focus our attention on assembling the core application logic.  This logic will have us retrieve a document from our vector store based on a user-provided question.  That question along with the document, added to provide context, will then be used to assemble a prompt which will then be sent to a model in order to generate a response. </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_application.png' width=900>
# MAGIC
# MAGIC </p>
# MAGIC In this notebook, we'll first walk through these steps one at a time so that we can wrap our head around what all is taking place.  We will then repackage the logic as a class object which will allow us to more easily encapsulate our work.  We will persist that object as a model within MLflow which will assist us in deploying the model in the last notebook associated with this accelerator.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install langchain==0.0.166 tiktoken==0.4.0 openai==0.27.6 faiss-cpu==1.7.4 typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import re
import time
import pandas as pd
import mlflow
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import BaseRetriever
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain import LLMChain

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: Explore Answer Generation
# MAGIC
# MAGIC To get started, let's explore how we will derive an answer in response to a user provide question.  We'll start by defining that question here:

# COMMAND ----------

# DBTITLE 1,Specify Question
question = "How to register a model on databricks?"

# COMMAND ----------

# MAGIC %md Using our vector store, assembled in the prior notebook, we will retrieve document chunks relevant to the question: 
# MAGIC
# MAGIC **NOTE** The OpenAI API key used by the OpenAIEmbeddings object is specified in an environment variable set during the earlier `%run` call to get configuration variables.

# COMMAND ----------

# DBTITLE 1,Retrieve Relevant Documents
# open vector store to access embeddings
embeddings = OpenAIEmbeddings(model=config['openai_embedding_model'])
vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])

# configure document retrieval 
n_documents = 5 # number of documents to retrieve 
retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # configure retrieval mechanism

# get relevant documents
docs = retriever.get_relevant_documents(question)
for doc in docs: 
  print(doc,'\n') 

# COMMAND ----------

# MAGIC %md We can now turn our attention to the prompt that we will send to the model.  This prompt needs to include placeholders for the *question* the user will submit and the document that we believe will provide the *context* for answering it.
# MAGIC
# MAGIC Please note that the prompt consists of multiple prompt elements, defined using [prompt templates](https://python.langchain.com/en/latest/modules/prompts/chat_prompt_template.html).  In a nutshell, prompt templates allow us to define the basic structure of a prompt and more easily substitute variable data into them to trigger a response.  The system message prompt shown here provides instruction to the model about how we want it to respond.  The human message template provides the details about the user-initiated request.
# MAGIC
# MAGIC The prompts along with the details about the model that will respond to the prompt are encapsulated within an [LLMChain object](https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html).  This object simply defines the basic structure for resolving a query and returning a response:

# COMMAND ----------

# DBTITLE 1,Define Chain to Generate Responses
# define system-level instructions
system_message_prompt = SystemMessagePromptTemplate.from_template(config['system_message_template'])

# define human-driven instructions
human_message_prompt = HumanMessagePromptTemplate.from_template(config['human_message_template'])

# combine instructions into a single prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# define model to respond to prompt
llm = ChatOpenAI(model_name=config['openai_chat_model'], temperature=config['temperature'])

# combine prompt and model into a unit of work (chain)
qa_chain = LLMChain(
  llm = llm,
  prompt = chat_prompt
  )

# COMMAND ----------

# MAGIC %md To actually trigger a response, we will loop through each of our docs from highest to lowest relevance and attempt to elicit a response.  Once we get a valid response, we'll stop.
# MAGIC
# MAGIC Please note, we aren't providing time-out handling or thoroughly validating the response from the model in this next cell.  We will want to make this logic more robust as we assemble our application class but for now we'll keep it simple to ensure the code is easy to read:

# COMMAND ----------

# DBTITLE 1,Generate a Response
# for each provided document
for doc in docs:

  # get document text
  text = doc.page_content

  # generate a response
  output = qa_chain.generate([{'context': text, 'question': question}])
 
  # get answer from results
  generation = output.generations[0][0]
  answer = generation.text

  # display answer
  if answer is not None:
    print(f"Question: {question}", '\n', f"Answer: {answer}")
    break

# COMMAND ----------

# MAGIC %md ##Step 2: Assemble Model for Deployment
# MAGIC
# MAGIC Having explored the basic steps involved in generating a response, let's wrap our logic in a class to make deployment easier.  Our class will be initialized by passing the LLM model definition, a vector store retriever and a prompt to the class.  The *get_answer* method will serve as the primary method for submitting a question and getting a response:

# COMMAND ----------

# DBTITLE 1,Define QABot Class
class QABot():


  def __init__(self, llm, retriever, prompt):
    self.llm = llm
    self.retriever = retriever
    self.prompt = prompt
    self.qa_chain = LLMChain(llm = self.llm, prompt=prompt)
    self.abbreviations = { # known abbreviations we want to replace
      "DBR": "Databricks Runtime",
      "ML": "Machine Learning",
      "UC": "Unity Catalog",
      "DLT": "Delta Live Table",
      "DBFS": "Databricks File Store",
      "HMS": "Hive Metastore",
      "UDF": "User Defined Function"
      } 


  def _is_good_answer(self, answer):

    ''' check if answer is a valid '''

    result = True # default response

    badanswer_phrases = [ # phrases that indicate model produced non-answer
      "no information", "no context", "don't know", "no clear answer", "sorry", 
      "no answer", "no mention", "reminder", "context does not provide", "no helpful answer", 
      "given context", "no helpful", "no relevant", "no question", "not clear",
      "don't have enough information", " does not have the relevant information", "does not seem to be directly related"
      ]
    
    if answer is None: # bad answer if answer is none
      results = False
    else: # bad answer if contains badanswer phrase
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    
    return result


  def _get_answer(self, context, question, timeout_sec=60):

    '''' get answer from llm with timeout handling '''

    # default result
    result = None

    # define end time
    end_time = time.time() + timeout_sec

    # try timeout
    while time.time() < end_time:

      # attempt to get a response
      try: 
        result =  qa_chain.generate([{'context': context, 'question': question}])
        break # if successful response, stop looping

      # if rate limit error...
      except openai.error.RateLimitError as rate_limit_error:
        if time.time() < end_time: # if time permits, sleep
          time.sleep(2)
          continue
        else: # otherwise, raiser the exception
          raise rate_limit_error

      # if other error, raise it
      except Exception as e:
        print(f'LLM QA Chain encountered unexpected error: {e}')
        raise e

    return result


  def get_answer(self, question):
    ''' get answer to provided question '''

    # default result
    result = {'answer':None, 'source':None, 'output_metadata':None}

    # remove common abbreviations from question
    for abbreviation, full_text in self.abbreviations.items():
      pattern = re.compile(fr'\b({abbreviation}|{abbreviation.lower()})\b', re.IGNORECASE)
      question = pattern.sub(f"{abbreviation} ({full_text})", question)

    # get relevant documents
    docs = self.retriever.get_relevant_documents(question)

    # for each doc ...
    for doc in docs:

      # get key elements for doc
      text = doc.page_content
      source = doc.metadata['source']

      # get an answer from llm
      output = self._get_answer(text, question)
 
      # get output from results
      generation = output.generations[0][0]
      answer = generation.text
      output_metadata = output.llm_output

      # assemble results if not no_answer
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        result['output_metadata'] = output_metadata
        break # stop looping if good answer
      
    return result

# COMMAND ----------

# MAGIC %md Now we can test our class using the objects instantiated earlier:

# COMMAND ----------

# DBTITLE 1,Test the QABot Class
# instantiate bot object
qabot = QABot(llm, retriever, chat_prompt)

# get response to question
qabot.get_answer(question) 

# COMMAND ----------

# MAGIC %md ##Step 3: Persist Model to MLflow
# MAGIC
# MAGIC With our bot class defined and validated, we can now persist it to MLflow.  MLflow is an open source repository for model tracking and logging.  It's deployed by default with the Databricks platform, making it easy for us to record models with it.
# MAGIC
# MAGIC While MLflow now [supports](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html) both OpenAI and LangChain model flavors, the fact that we've written custom logic for our bot application means that we'll need to make use of the more generic [pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models) model flavor.  This model flavor allows us to write a custom wrapper for our model that gives us considerable control over how our model responds when deployed through standard, MLflow-provided deployment mechanisms. 
# MAGIC
# MAGIC To create a custom MLflow model, all we need to do is define a class wrapper of type *mlflow.pyfunc.PythonModel*. The *__init__* method will initialize an instance of our *QABot* class and persist it to an class variable.  And a *predict* method will serve as the standard interface for generating a response.  That method will receive our inputs as a pandas dataframe but we can write the logic with the knowledge that it will only be receiving one user-provided question at a time:

# COMMAND ----------

# DBTITLE 1,Define MLflow Wrapper for Model
class MLflowQABot(mlflow.pyfunc.PythonModel):

  def __init__(self, llm, retriever, chat_prompt):
    self.qabot = QABot(llm, retriever, chat_prompt)

  def predict(self, context, inputs):
    questions = list(inputs['question'])

    # return answer
    return [self.qabot.get_answer(q) for q in questions]

# COMMAND ----------

# MAGIC %md We can then instantiate our model and log it to the [MLflow registry](https://docs.databricks.com/mlflow/model-registry.html) as follows:

# COMMAND ----------

# DBTITLE 1,Persist Model to MLflow
# instantiate mlflow model
model = MLflowQABot(llm, retriever, chat_prompt)

# persist model to mlflow
with mlflow.start_run():
  _ = (
    mlflow.pyfunc.log_model(
      python_model=model,
      extra_pip_requirements=['langchain==0.0.166', 'tiktoken==0.4.0', 'openai==0.27.6', 'faiss-cpu==1.7.4', 'typing-inspect==0.8.0', 'typing_extensions==4.5.0'],
      artifact_path='model',
      registered_model_name=config['registered_model_name']
      )
    )


# COMMAND ----------

# MAGIC %md If you are new to MLflow, you may be wondering what logging is doing for us.  If you navigate to the experiment associated with this notebook - look for the flask icon in the right-hand navigation of your Databricks environment to access the experiments - you can click on the latest experiment to see details about what was recorded with the *log_model* call. If you expand the model artifacts, you should see a *python_model.pkl* file that represents the pickled MLflowQABot model instantiated before.  It's this model that we retrieve when we (later) load our model into this or another environment:
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/bot_mlflow_log_model.PNG" width=1000>

# COMMAND ----------

# MAGIC %md The MLflow model registry provides mechanisms for us to manage our registered models as they move through a CI/CD workflow.  If we want to just push a model straight to production status (which is fine for a demo but not recommended in real-world scenarios), we can do this programmatically as follows:

# COMMAND ----------

# DBTITLE 1,Elevate Model to Production Status
# connect to mlflow 
client = mlflow.MlflowClient()

# identify latest model version
latest_version = client.get_latest_versions(config['registered_model_name'], stages=['None'])[0].version

# move model into production
client.transition_model_version_stage(
    name=config['registered_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md We can then retrieve the model from the registry and submit a few questions to verify the response:

# COMMAND ----------

# DBTITLE 1,Test the Model
# retrieve model from mlflow
model = mlflow.pyfunc.load_model(f"models:/{config['registered_model_name']}/Production")

# assemble question input
queries = pd.DataFrame({'question':[
  "How to read data with Delta Sharing?",
  "What are Delta Live Tables datasets?",
  "How to set up Unity Catalog?"
]})

# get a response
model.predict(queries)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
