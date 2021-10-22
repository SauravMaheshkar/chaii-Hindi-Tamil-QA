<div align = "center">
  <img src = "https://github.com/SauravMaheshkar/chaii-Hindi-Tamil-QA/blob/main/assets/Coffee%20Banner.png?raw=true">
</div>

My Model files and code for '**chaii - Hindi and Tamil Question Answering**' Kaggle competition organized by Google

<details>
  <summary>Initial Baseline Experiments</summary>

<br>

[Link to Weights and Biases 🔥 Interactive Dashboard](https://wandb.ai/sauravmaheshkar/chaii/reports/Baseline-Models-LB--Vmlldzo5NzYzMTE).

<br>

All models were taken from [**Huggingface Question Answering Models**](https://www.kaggle.com/sauravmaheshkar/huggingface-question-answering-models) trained using @rhtsingh's processed dataset [ [**External Data - MLQA, XQUAD Preprocessing**](https://www.kaggle.com/rhtsingh/external-data-mlqa-xquad-preprocessing) ] using huggingface/transformers inbuilt [**weights and biases logger**](https://docs.wandb.ai/guides/integrations/huggingface).

![](https://raw.githubusercontent.com/SauravMaheshkar/chaii-Hindi-Tamil-QA/500ff923d44525d25d28a7b299995200b36c76cd/assets/Evaluation%20Loss.svg)

### The Model Weights can be found [here](https://www.kaggle.com/sauravmaheshkar/chaiifinetunedbaselinemodels)

|**Name**                                                |**Training Loss**| **Evaluation Loss**         |
|-----------------------------------------------------|----------|-------------------|
|electra-base-squad2                                  |1.9823    |2.27 |
|distilbert-base-cased-distilled-squad                |1.1694    |1.31 |
|bert-base-cased-squad2                               |1.0992    |1.26  |
|distilbert-base-uncased-distilled-squad              |1.0642    |1.19 |
|bert-large-uncased-whole-word-masking-squad2         |0.9206    |1.02 |
|bert-large-uncased-whole-word-masking-finetuned-squad|0.9068    |1.01  |
|xlm-roberta-base-squad2                              |0.7908    |0.90 |
|distilbert-**multi**-finetuned-for-xqua-on-tydiqa        |0.7827    |0.89  |
|bert-**multi**-uncased-finetuned-xquadv1                 |0.7072    |0.93 |
|bert-**multi**-cased-finetuned-xquadv1                   |0.6517    |0.74 |
|bert-base-**multilingual**-cased-finetuned-squad         |0.6257    |0.73 |
|xlm-**multi**-roberta-large-squad2                       |0.6209    |0.74  |
|bert-**multi**-cased-finedtuned-xquad-tydiqa-goldp       |0.6156    |0.70 |
|roberta-large-squad2                                 |0.2488    |0.36 |
|roberta-base-squad2                                  |0.236     |0.35|
</details>
