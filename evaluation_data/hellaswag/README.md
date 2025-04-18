# Hellaswag Dataset
## Evaluation Adjustment
The HellaSwag dataset includes three splits: train, validation, and test. However, the test split does not provide ground truth labels.
To evaluate performance in our pipeline, we use the validation split as a substitute for testing and compute scores accordingly.

For official evaluation, submissions should follow the required format and be submitted to the HellaSwag [leaderboard](https://rowanzellers.com/hellaswag/).

## Submission Format Example
```python
import pandas as pd
import numpy as np
test_probs = np.random.randn(10003, 4)

test_probs_df = pd.DataFrame(test_probs, index=[f'test-{i}' for i in range(test_probs.shape[0])],
                           columns=['ending0', 'ending1', 'ending2', 'ending3'])
test_probs_df.index.name = 'annot_id'
test_probs_df.to_csv('examplesubmission.csv')
```
