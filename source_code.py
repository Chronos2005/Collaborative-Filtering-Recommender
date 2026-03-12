import numpy as np
import logging, csv

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logger initialised')


# Hyperparameters
KNN_NEIGHBORS      = 60     # number of most-similar neighbours used per prediction
SHRINKAGE_LAMBDA   = 5      # Bayesian shrinkage: n_ij / (n_ij + lambda)
SIG_WEIGHT_THRESH  = 0      # significance weighting threshold: min(n_ij / T, 1); 0 = disabled
SIM_AMPLIFICATION  = 1.6    # case amplification exponent applied to similarities
SIM_DENOM_REG      = 0.05   # small constant added to prediction denominator to damp noise
MIN_RATING         = 1.0    # minimum allowed predicted rating (ratings are {1,2,3,4,5})
MAX_RATING         = 5.0    # maximum allowed predicted rating


def load_data( filepath=None ):
    """
    Parse a CSV file of ratings into a list of tuples.

    Training format (4 cols):  user_id, item_id, rating, timestamp
    Test format    (3 cols):  user_id, item_id, timestamp

    Returns a list of (user_id, item_id, rating_or_None, timestamp) tuples.
    """
    entries = []
    with open( filepath, 'r' ) as f:
        for row in csv.reader( f ):
            uid, iid = int( row[0] ), int( row[1] )
            if len( row ) == 4:
                entries.append( (uid, iid, float( row[2] ), int( row[3] )) )
            elif len( row ) == 3:
                entries.append( (uid, iid, None, int( row[2] )) )

    logger.info( f'Parsed {len(entries)} records from {filepath}' )
    return entries


def train_model( train_data=None, n_users=None, n_items=None,
                 shrinkage_lambda=SHRINKAGE_LAMBDA, sig_weight_thresh=SIG_WEIGHT_THRESH ):
    """
    Build an item-item similarity model using adjusted cosine similarity.

    1.  Baseline Bias Model
        Compute user and item biases as simple deviations from global mean:
            b_u = mean(r_u) - mu   (user bias: harsh vs generous rater)
            b_i = mean(r_i) - mu   (item bias: popular vs unpopular item)
        
        Baseline prediction: b_ui = mu + b_u + b_i

    2.  Standard Adjusted Cosine Similarity (vectorised)
        Ratings are centered by user mean:

            z_ui = r_ui - mean_u

        sim(i, j)  =  sum_u [z_ui * z_uj]
                      ----------------------
                        ||z_i|| * ||z_j||

        Computed for all item pairs simultaneously using matrix multiplication:
            numerator   = Z^T @ Z
            denominator = outer product of per-column L2 norms

    3.  Empirical Bayes Shrinkage
        sim_adj(i, j) = sim(i, j) * n_ij / (n_ij + lambda)

        where n_ij = number of users who co-rated items i and j.
        This smoothly discounts similarity estimates based on few co-raters,
        unlike a hard threshold which creates a discontinuity.

    4.  Significance Weighting
        sim'(i, j) = sim(i, j) * min(n_ij / T, 1)

        where T = threshold (e.g., 50). Linearly penalises similarities
        based on fewer than T co-raters, reaching full weight at T.

    Returns: (user_biases, item_biases, similarity_matrix, global_mean)
    """
    # Infer dimensions from data if not provided (needed for cross-validation consistency)
    if n_users is None:
        n_users = max( row[0] for row in train_data )
    if n_items is None:
        n_items = max( row[1] for row in train_data )

    logger.info( f'Building model: {n_users} users, {n_items} items' )

    # Populate user-item rating matrix R; R[u][i] = 0 means unrated
    rating_matrix = np.zeros( (n_users, n_items), dtype=np.float64 )
    for uid, iid, rating, ts in train_data:
        rating_matrix[uid - 1][iid - 1] = rating

    # Binary mask and counts for vectorised operations
    has_rating = (rating_matrix != 0).astype( np.float64 )
    total_ratings = has_rating.sum()
    global_mean = rating_matrix.sum() / total_ratings if total_ratings > 0 else 0.0
    counts_per_user = has_rating.sum( axis=1 )
    counts_per_item = has_rating.sum( axis=0 )

    logger.info( f'Global mean rating: {global_mean:.4f}' )

    # ---- Baseline Bias Model ----
    # User bias: b_u = mean(r_u) - mu  (how much user deviates from global mean)
    user_means = np.where( counts_per_user > 0, 
                           rating_matrix.sum( axis=1 ) / np.maximum( counts_per_user, 1 ), 
                           global_mean )
    user_biases = user_means - global_mean

    # Item bias: b_i = mean(r_i) - mu  (how much item deviates from global mean)
    item_means = np.where( counts_per_item > 0,
                           rating_matrix.sum( axis=0 ) / np.maximum( counts_per_item, 1 ),
                           global_mean )
    item_biases = item_means - global_mean

    # ---- Standard Adjusted Cosine Similarity ----
    # Center ratings by user mean: z_ui = r_ui - mean_u  for rated entries, 0 otherwise
    centered = np.where( has_rating > 0, rating_matrix - user_means[:, np.newaxis], 0.0 )

    # ---- Vectorised Adjusted Cosine Similarity ----
    # sim(i, j) = sum_u[z_ui * z_uj] / (||z_i|| * ||z_j||)
    logger.info( 'Computing similarity matrix...' )
    with np.errstate( all='ignore' ):
        co_rater_counts = has_rating.T @ has_rating
        numerator = centered.T @ centered
    norms = np.linalg.norm( centered, axis=0 )
    norms[norms == 0] = 1e-10
    similarity_matrix = numerator / np.outer( norms, norms )
    np.nan_to_num( similarity_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0 )

    # ---- Empirical Bayes Shrinkage ----
    similarity_matrix *= co_rater_counts / (co_rater_counts + shrinkage_lambda)

    # ---- Significance Weighting ----
    # sim'(i,j) = sim(i,j) * min(n_ij / T, 1)  — linearly penalise if < T co-raters
    if sig_weight_thresh > 0:
        sig_weight = np.minimum( co_rater_counts / sig_weight_thresh, 1.0 )
        similarity_matrix *= sig_weight

    np.fill_diagonal( similarity_matrix, 0.0 )

    logger.info( f'Similarity matrix ready: {similarity_matrix.shape}' )
    return user_biases, item_biases, similarity_matrix, global_mean


def predict_ratings( test_data=None, similarity_matrix=None, user_biases=None,
                     item_biases=None, train_data=None, n_users=None, n_items=None,
                     global_mean=None,
                     knn_neighbors=KNN_NEIGHBORS, sim_amplification=SIM_AMPLIFICATION ):
    """
    Predict ratings using k-nearest-neighbour (KNN) with baseline bias model:

        pred(u, i) = mu + b_u + b_i  +  sum_j [ w_j * (r_uj - (mu + b_u + b_j)) ]
                                        -----------------------------------------
                                        sum_j |w_j|  +  c

    where:
        mu                                   (global mean rating)
        b_u                                  (user bias: harsh vs generous rater)
        b_i                                  (item bias: popular vs unpopular item)
        w_j  = sign(s_ij) * |s_ij|^p         (case-amplified similarity)
        r_uj                                 (user u's actual rating on neighbor item j)
        b_j                                  (neighbor item's bias)
        c    = SIM_DENOM_REG                 (denominator regularisation constant)

    The baseline mu + b_u + b_i captures:
        - Global rating tendency
        - User's rating tendency (harsh vs generous)
        - Item's popularity (good vs bad item)

    The neighbour term adds deviations from the baseline, weighted by similarity.

    Top-K selection uses np.argpartition for O(N) complexity instead of a full
    O(N log N) sort.  Only positively-correlated neighbours are retained.

    Fallback: if no positive neighbours exist, the baseline is returned.
    All predictions are clamped to [MIN_RATING, MAX_RATING].
    """
    if n_users is None:
        n_users = len( user_biases )
    if n_items is None:
        n_items = similarity_matrix.shape[0]

    # Reconstruct rating matrix for neighbour lookups
    rating_matrix = np.zeros( (n_users, n_items), dtype=np.float64 )
    for uid, iid, rating, ts in train_data:
        rating_matrix[uid - 1][iid - 1] = rating

    results = []

    for uid, iid, _, ts in test_data:
        u_idx, i_idx = uid - 1, iid - 1
        user_rated = np.nonzero( rating_matrix[u_idx] )[0]
        
        # Baseline prediction: mu + b_u + b_i
        baseline = global_mean + user_biases[u_idx] + item_biases[i_idx]

        if user_rated.size == 0:
            # Cold start: fallback to baseline
            prediction = baseline
        else:
            sims = similarity_matrix[i_idx][user_rated]

            # O(N) partial sort for top-K by absolute similarity
            if user_rated.size > knn_neighbors:
                part_idx = np.argpartition( np.abs( sims ), -knn_neighbors )[-knn_neighbors:]
            else:
                part_idx = np.arange( user_rated.size )

            # Discard negatively correlated neighbours (they introduce noise)
            top_idx = part_idx[sims[part_idx] > 0]

            if top_idx.size == 0:
                prediction = baseline
            else:
                nbr_items = user_rated[top_idx]
                nbr_sims = sims[top_idx]
                nbr_ratings = rating_matrix[u_idx][nbr_items]

                # Case amplification: sharpens contrast between strong/weak neighbours
                amplified = np.sign( nbr_sims ) * np.abs( nbr_sims ) ** sim_amplification

                # Deviation of each neighbour's rating from its baseline (mu + b_u + b_j)
                nbr_baselines = global_mean + user_biases[u_idx] + item_biases[nbr_items]
                nbr_deviations = nbr_ratings - nbr_baselines

                weighted_sum = np.dot( amplified, nbr_deviations )
                sim_total = np.sum( np.abs( amplified ) )

                # Regularised denominator damps noise when sum of similarities is small
                prediction = baseline + weighted_sum / (sim_total + SIM_DENOM_REG)

        # Round to nearest integer — training ratings are {1,2,3,4,5} so
        # aligning predictions to the discrete target space reduces MAE
        prediction = np.clip( np.round( prediction ), MIN_RATING, MAX_RATING )
        results.append( (uid, iid, prediction, ts) )

    logger.info( f'Generated {len(results)} predictions' )
    return results


def serialize_predictions( filepath=None, predictions=None ):
    """Export predicted ratings to CSV: user_id, item_id, predicted_rating, timestamp."""
    with open( filepath, 'w', newline='' ) as f:
        writer = csv.writer( f )
        for entry in predictions:
            writer.writerow( entry )

    logger.info( f'Wrote {len(predictions)} predictions to {filepath}' )


if __name__ == '__main__':

    logger.info( 'Item-Based Collaborative Filtering Recommender' )

    # Load datasets
    train = load_data( filepath='train_100k_withratings.csv' )
    test = load_data( filepath='test_100k_withoutratings.csv' )

    # ---- 5-Fold Cross-Validation ----
    # Partitions data into K folds; each fold is held out once for validation
    # while the model is trained on the remaining K-1 folds.
    # Reports mean MAE +/- std across folds.
    K_FOLDS = 5
    np.random.seed( 42 )
    shuffled = np.random.permutation( len(train) )

    sizes = np.full( K_FOLDS, len(train) // K_FOLDS, dtype=int )
    sizes[:len(train) % K_FOLDS] += 1
    folds, offset = [], 0
    for s in sizes:
        folds.append( shuffled[offset:offset + s] )
        offset += s

    logger.info( f'{K_FOLDS}-Fold CV: {len(train)} ratings, fold sizes: {[len(f) for f in folds]}' )

    total_users = max( e[0] for e in train )
    total_items = max( e[1] for e in train )
    mae_per_fold = []

    for k in range( K_FOLDS ):
        logger.info( f'--- Fold {k + 1} / {K_FOLDS} ---' )

        val_idx = folds[k]
        train_idx = np.concatenate( [folds[j] for j in range(K_FOLDS) if j != k] )

        train_fold = [train[i] for i in train_idx]
        val_fold = [train[i] for i in val_idx]

        fold_ubiases, fold_ibiases, fold_sim, fold_gmean = train_model(
            train_data=train_fold, n_users=total_users, n_items=total_items
        )

        val_queries = [(uid, iid, None, ts) for uid, iid, _, ts in val_fold]
        val_preds = predict_ratings( test_data=val_queries, similarity_matrix=fold_sim,
                                     user_biases=fold_ubiases, item_biases=fold_ibiases,
                                     train_data=train_fold,
                                     n_users=total_users, n_items=total_items,
                                     global_mean=fold_gmean )

        actual = np.array( [e[2] for e in val_fold] )
        predicted = np.array( [e[2] for e in val_preds] )
        mae = np.mean( np.abs( actual - predicted ) )
        mae_per_fold.append( mae )

        logger.info( f'Fold {k + 1} MAE: {mae:.6f}' )

    cv_mean = np.mean( mae_per_fold )
    cv_std = np.std( mae_per_fold )
    logger.info( f'*** {K_FOLDS}-Fold CV Result: MAE = {cv_mean:.6f} (+/- {cv_std:.6f}) ***' )

    # ---- Final Model on Full Training Set ----
    logger.info( 'Training final model on complete training set...' )
    user_biases, item_biases, sim_matrix, global_mean = train_model( train_data=train )

    predictions = predict_ratings( test_data=test, similarity_matrix=sim_matrix,
                                   user_biases=user_biases, item_biases=item_biases,
                                   train_data=train, global_mean=global_mean )

    serialize_predictions( filepath='submission.csv', predictions=predictions )
    logger.info( 'Submission file saved successfully' )