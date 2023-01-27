# Databricks notebook source

# MAGIC %md
# MAGIC # MLflow Classification Recipe Databricks Notebook
# MAGIC This notebook runs the MLflow Classification Recipe on Databricks and inspects its results.
# MAGIC

# COMMAND ----------

from mlflow.recipes import Recipe

r = Recipe(profile="databricks_staging")

r.run()