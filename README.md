https://drive.google.com/drive/folders/1qBovw44ooOrlT1yP_onUIVjUtQX2CEAq?usp=sharing  

the checkpoints 

to run the demo 
in the venv envi run :
```bash
python main.py
```

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
 
