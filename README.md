# Item-Based Collaborative Filtering on MovieLens 100K

A movie rating predictor built from first principles in **pure NumPy**: no SciPy, scikit-learn, Surprise or other recommender libraries. It uses **item-item collaborative filtering with adjusted cosine similarity** and predicts how a user would rate a film they haven't seen yet, using the ratings they gave to similar films.

On the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset (943 users, 1,682 movies, 100,000 ratings), it reaches a **mean absolute error (MAE) of 0.676** under 5-fold cross-validation. Cross-validation, training and prediction together take about **5 seconds** on a laptop.

## Results

5-fold cross-validation on the 90,570-rating training set (seed 42):

| Model                                   | MAE        |
|-----------------------------------------|------------|
| Predict the global mean                 | 0.945      |
| Baseline biases (μ + b<sub>u</sub> + b<sub>i</sub>) | 0.721      |
| **This model (item-KNN on top of the baseline)** | **0.676 ± 0.004** |

The neighbourhood model cuts error by about **6%** compared with the bias-only baseline and by **28%** compared with predicting the mean.

## How it works

### 1. Baseline bias model
Each rating is first explained by three simple effects:

```
b_ui = μ + b_u + b_i
```

- **μ** is the global mean rating.
- **b<sub>u</sub>** is the user bias: how much this user tends to rate above or below the mean.
- **b<sub>i</sub>** is the item bias: how much this film tends to be rated above or below the mean.

The similarity model then only has to predict the *deviation* from this baseline.

### 2. Adjusted cosine similarity (vectorised)
Ratings are centred on each user's mean, so that a harsh rater's 3 and a generous rater's 4 can mean the same thing. Similarity between every pair of items is then computed in one step with matrix multiplication:

```
sim(i, j) = (Zᵀ Z)_ij / (‖z_i‖ · ‖z_j‖)
```

This produces the full 1,682 × 1,682 similarity matrix in well under a second, without Python loops.

### 3. Shrinkage for low-evidence pairs
A similarity based on 3 shared raters is far less reliable than one based on 300. Each similarity is therefore scaled down by an empirical-Bayes shrinkage factor:

```
sim'(i, j) = sim(i, j) · n_ij / (n_ij + λ)
```

where `n_ij` is the number of users who rated both films. An optional hard significance-weighting threshold is also implemented for comparison.

### 4. K-nearest-neighbour prediction
For a (user, film) query:

1. From the films the user has rated, select the top-K most similar to the target film. This uses `np.argpartition`, which runs in O(N) time instead of the O(N log N) needed for a full sort.
2. Drop negatively correlated neighbours, because they added noise in testing.
3. Apply **case amplification** (`w = sim^p`) so that strong neighbours count for more than weak ones.
4. Combine the neighbours' deviations from their own baselines, weighted by similarity, and add the result to the target's baseline.
5. Clamp the prediction to the 1–5 range and round it to the nearest whole star (ratings are integers, so rounding lowers MAE).

Users or films with no usable neighbours fall back to the baseline prediction.

### 5. Hyperparameter tuning
The hyperparameters were chosen by grid search with 5-fold cross-validation. The search covered K ∈ {20…80}, λ ∈ {5…100} and amplification p ∈ {1.0…2.0}:

| Parameter            | Value | Role                                         |
|----------------------|-------|----------------------------------------------|
| `KNN_NEIGHBORS`      | 60    | Neighbours used per prediction               |
| `SHRINKAGE_LAMBDA`   | 5     | Strength of the low-co-rater penalty         |
| `SIM_AMPLIFICATION`  | 1.6   | Exponent that sharpens strong vs weak neighbours |
| `SIM_DENOM_REG`      | 0.05  | Damps predictions when total similarity is small |

## Running it

**Requirements:** Python 3.8+ and NumPy.

```bash
pip install numpy
python source_code.py
```

The script expects two CSV files in the working directory. Both are splits of MovieLens 100K:

| File                            | Format                                   |
|---------------------------------|------------------------------------------|
| `train_100k_withratings.csv`    | `user_id, item_id, rating, timestamp`    |
| `test_100k_withoutratings.csv`  | `user_id, item_id, timestamp`            |

The script will:
1. Run 5-fold cross-validation and log the MAE for each fold and overall.
2. Retrain on the full training set.
3. Write predictions for the test set to `submission.csv`.

## Project structure

```
source_code.py   # Loads data, trains the model, predicts, cross-validates and writes the CSV
```

The core is three functions:
- `train_model()` computes the biases and the shrunk similarity matrix.
- `predict_ratings()` makes the KNN predictions on top of the baseline.
- `load_data()` and `serialize_predictions()` handle CSV input and output.

## Skills demonstrated

- **Recommender systems:** neighbourhood-based collaborative filtering, baseline predictors, similarity shrinkage and case amplification.
- **Numerical computing:** vectorised linear algebra in NumPy, plus attention to memory use and algorithmic complexity (O(N) partial sort for top-K).
- **Experimental rigour:** K-fold cross-validation, comparison against baselines and systematic hyperparameter search.
- **Implementation from scratch:** every step is written by hand rather than called from a library.

## Dataset

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. https://doi.org/10.1145/2827872
