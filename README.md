# Path-based Self-Supervised Pretraining for Knowledge Graph Completion

The code for PC-SSP.

### Path Generation

Run path_produce.py under the generate_path dectionary to generate 2-hop paths for P2E and5-hop paths for P2P:

```markup
python path_produce.py
```

### Pretraining

Run main.py to pretrain specific model: P2E/P2P/Joint:

```markup
python main.py --mode P2E
```

The hyperparameter settings can be referred to in the following table:

<img width="1120" height="471" alt="image" src="https://github.com/user-attachments/assets/76f40f46-4200-4b5f-a721-470c71073464" />

### Link Prediction

The entity and relation embeddings learned in the pretraining stage can be used by subsequent KGC models, which include:

**ConvE, InteractE, RotatE, TuckER, FieldE** and so on.
