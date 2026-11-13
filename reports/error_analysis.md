
# CIFAR-10 Error Analysis

## Model Evaluated
- Model: Augmented CNN
- Dataset: CIFAR-10
- Test Images: 10,000
- Test Accuracy: 75.11%

## 1. Frequently Confused Classes

| Actual Class | Predicted Class | Incorrect Predictions |
|---|---|---:|
| Cat | Dog | 237 |
| Cat | Frog | 131 |
| Bird | Frog | 130 |
| Dog | Cat | 121 |
| Bird | Dog | 101 |

## 2. Prediction Confidence Analysis

| Error Type | Incorrect Predictions | Average Confidence |
|---|---:|---:|
| Cat -> Dog | 237 | 59.84% |
| Cat -> Frog | 131 | 67.40% |

### Observations

- Cat-to-dog confusion was the most frequent error.
- The model incorrectly classified 237 cats as dogs.
- Cat-to-frog predictions had a higher average confidence
  of 67.40%.
- The model can be confidently wrong when classifying
  visually challenging images.

## 3. Baseline vs Augmented Observations

- Baseline test accuracy: 72.04%.
- Augmented test accuracy: 75.11%.
- Test accuracy improved by 3.07 percentage points.
- Cat-to-dog errors increased from 220 to 237.
- Dog-to-cat errors decreased from 136 to 121.
- Bird-to-frog confusion appeared among the augmented
  model's most frequent errors.

## 4. Conclusion

Data augmentation improved the overall performance
of the CNN.

However, the model still struggles to distinguish
some visually challenging classes, especially cats,
dogs, and birds.

Further improvements could include a deeper CNN,
additional training experiments, and more advanced
data augmentation techniques.

The error analysis demonstrates that improved overall
accuracy does not guarantee better predictions for
every individual class.
