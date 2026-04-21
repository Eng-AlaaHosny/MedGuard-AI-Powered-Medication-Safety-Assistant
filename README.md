https://drive.google.com/drive/folders/1qBovw44ooOrlT1yP_onUIVjUtQX2CEAq?usp=sharing  

the checkpoints 

to run the demo 
in the venv envi run :
```bash
python main.py
```
the last training results without fusing the knowledge graph 
( checkpoint file is named best_model_3heads.pt in the google drive link )
<img width="1035" height="222" alt="image" src="https://github.com/user-attachments/assets/f0ccc765-0715-4aff-993e-83bd12edb712" />


Traing with the KG 
<img width="872" height="401" alt="image" src="https://github.com/user-attachments/assets/1652d965-434f-4304-971c-bab2762d825e" />

more training with 10 epoch 
<img width="995" height="293" alt="image" src="https://github.com/user-attachments/assets/e4675d4e-8aee-4978-b103-0516143e650e" />


```
MEDGUARD/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── api/
│   │       ├── __init__.py
│   │       └── routes.py
│   │
│   ├── data/
│   │   ├── __pycache__/
│   │   ├── DDICorpus/
│   │   ├── drugbank_full.xml
│   │   ├── DB_compounds_lipinski.csv
│   │   ├── drugbank_processor.py
│   │   ├── drugbank.db
│   │   ├── kg_embeddings.pkl
│   │   ├── knowledge_graph.pkl
│   │   ├── lipinski_processor.py
│   │   └── preprocessor.py
│   │
│   ├── knowledge_graph/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   └── kg_builder_full.py
│   │
│   ├── models/
│   │   ├── __pycache__/
│   │   ├── checkpoints/
│   │   ├── __init__.py
│   │   ├── medguard_model.py
│   │   └── trainer.py
│   │
│   ├── static/
│   │   └── demo.html
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── __init__.py
│   │
│   └── __pycache__/
│
├── frontend/
│
├── venv/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/
│   └── pyvenv.cfg
│
├── .env
├── main.py
├── .gitignore

```
 
