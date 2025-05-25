In here there are scripts that accurately predict the energy usage of over 1,000 buildings over a three-year timeframe, based on information about the building themselves
and the observed weather in the correspoding areas, that can be found here: [https://www.kaggle.com/competitions/ashrae-energy-prediction/data]. To run the scripts:


**Ray**

```bash
python3 ashrae_ray.py
```

**Spark**

```bash
spark-submit --packages "ch.cern.sparkmeasure:spark-measure_2.12:0.23,graphframes:graphframes:0.8.3-spark3.5-s_2.12" ashrae_spark.py <num_executors>
```
