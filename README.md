# Neural Network using Single Layer NN

## Επισκόπηση

* Single-layer Neural Network για addition
* Model: `prediction = w1 * a + w2 * b + bias`
* Στόχος: w1≈1, w2≈1, bias≈0

## Δομή Project

```
Neural-Net/
├── README.md
├── inference.py       # Interactive predictions
├── DATA/
│   ├── dataset.txt    # Training dataset
│   └── weights.json   # Saved model weights
└── scripts/
    ├── dataset-generator.py
    └── train_model.py
```

## Απαιτήσεις

* Python ≥ 3.6
* Μόνο standard library

## Γρήγορη Εκκίνηση

**1. Δημιουργία dataset**

```bash
python scripts/dataset-generator.py
```

**2. Εκπαίδευση Neural Network**

```bash
python scripts/train_model.py
```

* Training: 10,000 epochs, gradient descent
* Outputs: `DATA/weights.json`

**3. Inference**

```bash
python inference.py
```

* Interactive: `Enter 'a + b' : 15 + 27` → `Neural Network: 42`

## Λεπτομέρειες Neural Network

* Input: (a, b)
* Output: Predicted sum
* Loss: `(prediction - actual_sum)²`
* Backprop: gradient descent
* Stop όταν loss < 1e-10

## Χαρακτηριστικά

* Gradient Clipping
* Adaptive Learning Rate
* Real-time loss visualization
* Accuracy testing

## Αναμενόμενα Αποτελέσματα

* Training Accuracy: 100%
* Final Weights: w1 ≈ 1, w2 ≈ 1, bias ≈ 0
* Λειτουργεί για αριθμούς εντός του training range

## Επίλυση Προβλημάτων

* `No DATA/weights.json found!` → Train πρώτα
* `No DATA/dataset.txt found!` → run dataset-generator.py πρώτα
* Χαμηλή ακρίβεια σε μεγάλους αριθμούς → Expand dataset



