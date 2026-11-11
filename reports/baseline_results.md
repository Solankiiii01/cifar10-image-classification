
# CIFAR-10 Baseline Experiment Results

## Model
- Architecture: Custom CNN
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 20
- Training Images: 45,000
- Validation Images: 5,000
- Test Images: 10,000
- Data Augmentation: Disabled

## Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | 72.04% |
| Precision (Macro) | 0.7218 |
| Recall (Macro) | 0.7204 |
| F1-score (Macro) | 0.7201 |

## Observations

- Training accuracy reached 97.39%.
- Best validation accuracy was 72.76%.
- Test accuracy was 72.04%.
- Significant overfitting was observed.
- Automobile had the highest recall (85.90%).
- Cat had the lowest recall (52.50%).

## Conclusion

The baseline CNN achieved 72.04% test accuracy.
The difference between training and test performance
indicates overfitting. Data augmentation will be
investigated in the next experiment to improve
generalization.
