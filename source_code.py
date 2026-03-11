import numpy as np
import logging
import csv

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logger initialised')

# Hyperparameters
KNN_NEIGHBORS = 70
SIG_WEIGHT_THRESHOLD = 2
MIN_RATING, MAX_RATING = 0.5, 5.0
SIM_AMPLIFICATION = 1.6       # Sharpen strong neighbours by raising sims to this power
ITEM_SHRINKAGE = 3            # Shrink item means toward global mean for sparse items


def load_data( filepath=None ):
    """
    Parse a CSV file into a list of rating tuples.

    Args:
        filepath (str): Path to the input CSV file.

    Returns:
        list: Tuples of (user_id, item_id, rating, timestamp).
              Rating is None for test entries.
    """

    entries = []

    with open( filepath, 'r' ) as f:
        for row in csv.reader( f ):
            uid, iid = int( row[0] ), int( row[1] )
            # 4-column rows contain a rating; 3-column rows are test data
            if len( row ) == 4:
                entries.append( (uid, iid, float( row[2] ), int( row[3] )) )
            elif len( row ) == 3:
                entries.append( (uid, iid, None, int( row[2] )) )

    logger.info( f'Parsed {len(entries)} records from {filepath}' )
    return entries


def train_model( train_data=None, n_users=None, n_items=None ):
    """
    Build an item-item similarity model via adjusted cosine similarity.

    Adjusted cosine formula:
    sim(i, j) = sum_u z_ui * z_uj / (norm_i * norm_j)

    Where z_ui = (r_ui - mean_u) / std_u (Z-Score normalization).

    Significance weighting:
    sim_adj(i, j) = sim(i, j) * min(|co_raters|, beta) / beta

    Args:
        train_data (list): Rating tuples for training.
        n_users (int, optional): Number of users in the dataset.
        n_items (int, optional): Number of items in the dataset.

    Returns:
        tuple: (user_means, similarity_matrix, item_biases, global_mean)
    """

    if n_users is None:
        n_users = max( row[0] for row in train_data )
    if n_items is None:
        n_items = max( row[1] for row in train_data )

    logger.info( f'Building model: {n_users} users, {n_items} items' )

    # Populate the user-item rating matrix
    rating_matrix = np.zeros( (n_users, n_items), dtype=np.float64 )
    for uid, iid, rating, ts in train_data:
        rating_matrix[uid - 1][iid - 1] = rating

    # Mean rating per user, only over rated entries
    has_rating = (rating_matrix != 0).astype( np.float64 )
    counts_per_user = has_rating.sum( axis=1 )
    counts_per_user[counts_per_user == 0] = 1
    user_means = rating_matrix.sum( axis=1 ) / counts_per_user

    logger.info( f'Mean of user averages: {user_means.mean():.4f}' )

    # Z-Score normalize the rating matrix (handles user scale differences)
    centered = rating_matrix.copy()
    user_stds = np.ones( n_users )
    for u in range( n_users ):
        mask = centered[u] != 0
        if mask.sum() > 1:
            user_stds[u] = np.std( rating_matrix[u][mask] )
            if user_stds[u] == 0:
                user_stds[u] = 1.0
        centered[u][mask] = (centered[u][mask] - user_means[u]) / user_stds[u]

    # Compute adjusted cosine similarity via vectorised operations
    logger.info( 'Computing similarity matrix...' )
    co_rater_counts = has_rating.T @ has_rating
    dot_products = centered.T @ centered
    norms = np.linalg.norm( centered, axis=0 )
    norms[norms == 0] = 1e-10
    norm_products = np.outer( norms, norms )
    similarity_matrix = dot_products / norm_products
    np.nan_to_num( similarity_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0 )

    # Significance weighting: discount pairs with few co-raters
    sig_weights = np.minimum( co_rater_counts, SIG_WEIGHT_THRESHOLD ) / SIG_WEIGHT_THRESHOLD
    similarity_matrix *= sig_weights

    # Prevent an item from being its own neighbour
    np.fill_diagonal( similarity_matrix, 0.0 )
    logger.info( f'Similarity matrix ready: {similarity_matrix.shape}' )

    # Item biases with shrinkage toward global mean
    global_mean = user_means.mean()
    item_counts = has_rating.sum( axis=0 )
    item_sums = rating_matrix.sum( axis=0 )
    shrunk_item_means = (item_sums + ITEM_SHRINKAGE * global_mean) / (item_counts + ITEM_SHRINKAGE)
    item_biases = shrunk_item_means - global_mean

    return user_means, similarity_matrix, item_biases, global_mean


def predict_ratings( test_data=None, similarity_matrix=None, user_means=None, train_data=None, n_users=None, n_items=None, item_biases=None, global_mean=None ):
    """
    Generate predicted ratings using item-based collaborative filtering with KNN.

    Baseline-integrated prediction formula:
    pred(u, i) = b_ui + sum_j[ sim(i,j)^a * (r_uj - b_uj) ] / sum_j[ sim(i,j)^a ]

    Where b_ui = global_mean + user_bias + item_bias.
    Neighbours' deviations from their own baselines are weighted by
    amplified similarity, then added to the target item's baseline.

    Args:
        test_data (list): Entries to predict (user_id, item_id, None, timestamp).
        similarity_matrix (np.ndarray): Item-item similarity matrix.
        user_means (np.ndarray): Mean rating per user.
        train_data (list): Training data for building the rating lookup.
        n_users (int, optional): Number of users.
        n_items (int, optional): Number of items.
        item_biases (np.ndarray): Shrunk item bias per item.
        global_mean (float): Global average rating.

    Returns:
        list: Tuples of (user_id, item_id, predicted_rating, timestamp).
    """

    if n_users is None:
        n_users = len( user_means )
    if n_items is None:
        n_items = similarity_matrix.shape[0]

    # Reconstruct rating matrix for fast lookup
    rating_matrix = np.zeros( (n_users, n_items), dtype=np.float64 )
    for uid, iid, rating, ts in train_data:
        rating_matrix[uid - 1][iid - 1] = rating

    results = []

    for uid, iid, _, ts in test_data:
        u_idx, i_idx = uid - 1, iid - 1
        user_rated = np.nonzero( rating_matrix[u_idx] )[0]

        # Bias-corrected baseline for fallback
        user_bias = user_means[u_idx] - global_mean
        baseline = global_mean + user_bias + item_biases[i_idx]

        if user_rated.size == 0:
            prediction = baseline
        else:
            sims = similarity_matrix[i_idx][user_rated]

            # Select top-K neighbours by absolute similarity
            if user_rated.size > KNN_NEIGHBORS:
                top_idx = np.argsort( np.abs( sims ) )[-KNN_NEIGHBORS:]
            else:
                top_idx = np.arange( user_rated.size )

            # Keep only positively correlated neighbours
            top_idx = top_idx[sims[top_idx] > 0]

            if top_idx.size == 0:
                prediction = baseline
            else:
                nbr_items = user_rated[top_idx]
                nbr_sims = sims[top_idx]
                nbr_ratings = rating_matrix[u_idx][nbr_items]

                # Baseline-integrated: weight neighbour deviations from their baselines
                amplified = np.sign( nbr_sims ) * np.abs( nbr_sims ) ** SIM_AMPLIFICATION

                # Each neighbour has its own baseline: b_uj = mu + b_u + b_j
                nbr_baselines = global_mean + user_bias + item_biases[nbr_items]
                nbr_deviations = nbr_ratings - nbr_baselines

                weighted_sum = np.dot( amplified, nbr_deviations )
                sim_total = np.sum( amplified )

                if sim_total == 0:
                    prediction = baseline
                else:
                    prediction = baseline + weighted_sum / sim_total

        prediction = np.clip( prediction, MIN_RATING, MAX_RATING )
        results.append( (uid, iid, round( prediction, 4 ), ts) )

    logger.info( f'Generated {len(results)} predictions' )
    return results


def serialize_predictions( filepath=None, predictions=None ):
    """
    Export predicted ratings to a CSV file.

    Args:
        filepath (str): Destination CSV file path.
        predictions (list): Predicted rating tuples to write.
    """

    with open( filepath, 'w', newline='' ) as f:
        writer = csv.writer( f )
        for entry in predictions:
            writer.writerow( entry )

    logger.info( f'Wrote {len(predictions)} predictions to {filepath}' )


if __name__ == '__main__':

    logger.info( 'Item-Based Collaborative Filtering Recommender' )

    train = load_data( filepath='train_100k_withratings.csv' )
    test = load_data( filepath='test_100k_withoutratings.csv' )

    # K-Fold Cross-Validation
    K_FOLDS = 5
    np.random.seed( 42 )
    shuffled = np.random.permutation( len(train) )

    # Partition shuffled indices into K roughly equal folds
    sizes = np.full( K_FOLDS, len(train) // K_FOLDS, dtype=int )
    sizes[:len(train) % K_FOLDS] += 1
    folds = []
    offset = 0
    for s in sizes:
        folds.append( shuffled[offset:offset + s] )
        offset += s

    logger.info( f'{K_FOLDS}-Fold CV: {len(train)} ratings, fold sizes: {[len(f) for f in folds]}' )

    total_users = max( e[0] for e in train )
    total_items = max( e[1] for e in train )

    mae_per_fold = []

    for k in range( K_FOLDS ):
        logger.info( f'--- Fold {k + 1} / {K_FOLDS} ---' )

        # Hold out fold k for validation, use the rest for training
        val_idx = folds[k]
        train_idx = np.concatenate( [folds[j] for j in range(K_FOLDS) if j != k] )

        train_fold = [train[i] for i in train_idx]
        val_fold = [train[i] for i in val_idx]

        fold_means, fold_sim, fold_biases, fold_gmean = train_model( train_data=train_fold, n_users=total_users, n_items=total_items )

        val_queries = [(uid, iid, None, ts) for uid, iid, _, ts in val_fold]
        val_preds = predict_ratings( test_data=val_queries, similarity_matrix=fold_sim,
                                     user_means=fold_means, train_data=train_fold,
                                     n_users=total_users, n_items=total_items,
                                     item_biases=fold_biases, global_mean=fold_gmean )

        # Mean Absolute Error for this fold
        actual = np.array( [e[2] for e in val_fold] )
        predicted = np.array( [e[2] for e in val_preds] )
        mae = np.mean( np.abs( actual - predicted ) )
        mae_per_fold.append( mae )

        logger.info( f'Fold {k + 1} MAE: {mae:.6f}' )

    cv_mean = np.mean( mae_per_fold )
    cv_std = np.std( mae_per_fold )
    logger.info( f'*** {K_FOLDS}-Fold CV Result: MAE = {cv_mean:.6f} (+/- {cv_std:.6f}) ***' )

    # Train final model on all available data
    logger.info( 'Training final model on complete training set...' )
    user_means, sim_matrix, item_biases, global_mean = train_model( train_data=train )

    predictions = predict_ratings( test_data=test, similarity_matrix=sim_matrix,
                                   user_means=user_means, train_data=train,
                                   item_biases=item_biases, global_mean=global_mean )

    serialize_predictions( filepath='submission.csv', predictions=predictions )

    logger.info( 'Submission file saved successfully' )

