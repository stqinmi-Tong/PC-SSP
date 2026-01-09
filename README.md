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

![](Path-based%20Self-Supervised%20Pretraining%20for%20Knowledge%20Graph%20Completion_md_files/9a11fe50-ed5c-11f0-acfa-87b4eaab79a9_20260109211051.jpeg?v=1&type=image&token=V1%253AAlpnu2awwlMCK2OAqIaPVFpN77ly_0tD9KTEUzopo6Q)

### Link Prediction

The entity and relation embeddings learned in the pretraining stage can be used by subsequent KGC models, which include:

**ConvE, InteractE, RotatE, TuckER, FieldE** and so on.
