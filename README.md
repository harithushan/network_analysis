# network_analysis
network analysis on Disease-Symptom Dataset which has 773 Unique Diseases and 377 One-Hot Encoded Symptoms with 246,000 samples
## About Dataset
##Description
The dataset contains disease names along with the symptoms faced by the respective patient. There are a total of 773 unique diseases and 377 symptoms, with ~246,000 rows. The dataset was artificially generated, preserving Symptom Severity and Disease Occurrence Possibility.

##Interpretation Info
Several distinct groups of symptoms might all be indicators of the same disease. There may even be one single symptom contributing to a disease in a row or sample. This is an indicator of a very high correlation between the symptom and that particular disease.
A larger number of rows for a particular disease corresponds to its higher probability of occurrence in the real world. Similarly, in a row, if the feature vector has the occurrence of a single symptom, it implies that this symptom has more correlation to classify the disease than any one symptom of a feature vector with multiple symptoms in another sample.

Dataset link
https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset

```cmd
python -m venv venv
```
```cmd
venv\Scripts\activate
```

```cmd
pip install -r requirements.txt
```
```cmd
python data_processor.py
```

```cmd
streamlit run dashboard.py
```
