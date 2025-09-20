# network_analysis
network analysis on Disease-Symptom Dataset which has 773 Unique Diseases and 377 One-Hot Encoded Symptoms with 246,000 samples

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