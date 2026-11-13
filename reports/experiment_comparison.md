
# CIFAR-10 Experiment Comparison

## Experiments

| Setting | Baseline | Augmented |
|---|---|---|
| Architecture | Custom CNN | Custom CNN |
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.001 |
| Batch Size | 64 | 64 |
| Epochs | 20 | 20 |
| Data Augmentation | Disabled | Enabled |

## Test Results

| Metric | Baseline | Augmented |
|---|---:|---:|
| Accuracy | 72.04% | 75.11% |
| Precision | 0.7218 | 0.7592 |
| Recall | 0.7204 | 0.7511 |
| F1-score | 0.7201 | 0.7485 |

## Validation Results

| Metric | Baseline | Augmented |
|---|---:|---:|
| Best Validation Accuracy | 72.76% | 75.60% |
| Final Training Accuracy | 97.39% | 75.56% |

## Improvement

- Test accuracy improved by 3.07 percentage points.
- Precision improved by 0.0374.
- Recall improved by 0.0307.
- F1-score improved by 0.0284.

## Observations

1. Data augmentation improved overall test performance.
2. The baseline model showed significant overfitting.
3. The augmented model showed a much smaller training-validation gap.
4. Automobile and ship achieved strong F1-scores in the augmented experiment.
5. Cat and bird remain challenging classes.

## Conclusion

The augmented CNN achieved 75.11% test accuracy,
compared with 72.04% for the baseline CNN.

Random cropping and horizontal flipping helped
improve generalization on CIFAR-10.

The augmented model performed better overall.
