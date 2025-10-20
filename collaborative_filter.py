import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader
from surprise import SVD
from collections import defaultdict

# sample data: (user_id, item_id, rating)
# rating system: 5-point scale system

train_data = pd.read_csv('data/UKFS_testdata.csv', nrows=20)

print(f"Unique users: {train_data['user_id'].nunique()}")
print(f"Unique items: {train_data['item_id'].nunique()}")
print(f"Rating range: {train_data['rating'].min()} to {train_data['rating'].max()}")

if train_data['item_id'].nunique() < 5:
    print("Warning: Not enough unique items for meaningful recommendations.")

def load_data_correctly():
    df = pd.read_csv('data/UKFS_testdata.csv')

    print("Actual columns in file: ", df.columns.tolist())

    if 'user_id' not in df.columns and len(df.columns) >= 3:
        df.columns = ['user_id', 'item_id', 'rating']

    df = df.dropna(subset=['user_id', 'item_id', 'rating'])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    return df

train_data = load_data_correctly()

reader = Reader(rating_scale = (0, 2)) # reader is needed to parse the dataframe. 
dataset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']], reader)
trainset = dataset.build_full_trainset()

reader = Reader(rating_scale=(0, 2))
dataset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']], reader)
trainset = dataset.build_full_trainset()

# using the SVD algo.
model = SVD()
model.fit(trainset)
print("Model trained successfully!")

# function to get top-n recommendatiens.

def get_top_recommendations(user_id, trainset, model, data, n=5):
    print(f"\nGenerating recommendations for user: {user_id}")
    print(f" Total items in trainset: {len(trainset.all_items())}")

    # get a list of all items.
    all_items = trainset.all_items()

    # get user-rated items.
    # user_items = set([trainset.to_raw_iid(item_id) for [user, item_id] in trainset.ur[trainset.to_inner_uid(user_id)]])
    user_inner_id = trainset.to_inner_uid(user_id)

    user_items = set()
    for user, item_id in trainset.ur[user_inner_id]:
        if isinstance(item_id, int):
            raw_item_id = trainset.to_raw_iid(item_id)
            user_items.add(raw_item_id)


    # predictions.
    predictions = []
    for item_id in all_items:
        raw_item_id = trainset.to_raw_iid(item_id)
        if raw_item_id not in user_items: # predict for the first time only,
            pred = model.predict(user_id, raw_item_id)
            predictions.append((raw_item_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True) # sort by estimated ratings.

    return predictions[:n] # returns top-n recs.

print("\nGenerate recommendations for each user: ")

# CAP: Test only the first "n" users
user_count = 0

for user_id in train_data['user_id']:
    if user_count >= 20:
        break
    recommendations = get_top_recommendations(user_id, trainset, model, train_data, n=1)
    print(f"\nTop Recommnedations for {user_id}")
    for item, rating in recommendations:
        print(f"- {item}: predicted rating = {rating:.3f}")
    user_count += 1

print(f"\nCapped to the first {user_count} users.")


def create_temporal_split(interactions_df, test_weeks=2, validation_weeks=1):
    # This splits interactions by time to simulate real-world usage.

    # sort interations by timestamps.
    interactions_sorted = interactions_df.sort_values('timestamp')

    # calc. for cut of dates.
    max_date = interactions_sorted['timestamp'].max()
    test_start = max_date - pd.Timedelta(weeks=test_weeks)
    validation_start = test_start - pd.Timedelta(weeks=validation_weeks)

    # split the data.
    train_data = interactions_sorted[interactions_sorted['timestamp'] < validation_start]
    validation_data = interactions_sorted[(
        interactions_sorted['timestamp'] >= validation_start) & (interactions_sorted['timestamp'] < test_start)]

    test_data = interactions_sorted[interactions_sorted['timestamp'] >= test_start]

    print(f"training interactions: {len(train_data)}")
    print(f"validation interactions: {len(validation_data)}")
    print(f"test interactions: {len(test_data)}")

    if len(test_data) < 1000:
        print("Warning: Test set might be too small and it might invalidate the results.")

    return train_data, validation_data, test_data

def calc_precision_at_k(recommendations, actual_interactions, k=10):
    # calculates precision at k. what fraction of recommended items were actually relevant?

    """
    args: 
        recommendations: List of recommended items IDs, ordered by relevance
        actual_interactions: Set of item IDs that the user actually interacted with.
        k: Number of top recommendations to consider.
    
    returns:
        float between 0 and 1, where 1 means all recommended items were relevant. 
    
    """
    top_k = recommendations[:k]

    relevant_and_recommended = set(top_k) & set(actual_interactions)

    if len(top_k) == 0:
        return 0.0
    
    return len(relevant_and_recommended) / len(top_k)

def calc_recall_at_k(recommendations, actual_interactions, k=10):
    # calculates recall at k. what fraction of relevant items were recommended?

    """
    args: 
        recommendations: List of recommended items IDs, ordered by relevance
        actual_interactions: Set of item IDs that the user actually interacted with.
        k: Number of top recommendations to consider.
    
    returns:
        float between 0 and 1, where 1 means all relevant items were recommended. 

        This tells you how complete your recommendations are. You might have a high precision
        but if your recall is low, it means you're missing a lot of relevant items.
    """

    top_k = recommendations[:k]

    relevant_and_recommended = set(top_k) & set(actual_interactions)

    if len(actual_interactions) == 0:
        return 0.0
    
    return len(relevant_and_recommended) / len(actual_interactions)


def calc_f1_score(precision, recall):
    # calculates the harmonic mean of precision and recall.

    """ args:
        precision: Precision value
        recall: Recall value
    
    returns:
        float between 0 and 1, where 1 means perfect precision and recall.

        the f1 score balances both metrics, you can't get a high f1 score without both a good
        precision score and a good recall score.
    """

    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# implementing Mean Average Precision.
# this gives a more nuanced view of recommendation quality 
# by considering the rank of relevant items and putting those first.
# It's more sophisticated because it considers the order of recommendations.

def calc_average_precision(recommendations, actual_interactions, k=10):
    top_k = recommendations[:k]

    if len(actual_interactions) == 0:
        return 0.0

    precision_sum = 0.0
    num_relevant_found = 0

    for i, item in enumerate(top_k):
        if item in actual_interactions:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1) # precision at position.
            precision_sum += precision_at_i

    if num_relevant_found == 0:
        return 0.0

    return precision_sum / min(len(actual_interactions), k)

def calc_mean_avg_precision(all_recommendations, all_actual_interactions, k=10):
    # creates and calculates MAP for all users.
    # considering both relevance and ranking quality.

    avg_precisions = []

    for user_recs, user_actual in zip(all_recommendations, all_actual_interactions):
        ap = calc_average_precision(user_recs, user_actual, k)
        avg_precisions.append(ap)

    return np.mean(avg_precisions)


class RecommenderEvaluator:
    # a comprehensive eval framework, this class encapsulates all the eval logic
    # so you can easily evaluate different models consistently.

    def __init__(self, k_values=[5, 10, 20]):
        """
            k_values: differenct cutoff points to evaluate.

            what they see immediately (top_5) or what they see if they scroll down (top_20)
        """

        self.k_values = k_values
        self.results = {}

    def evaluate(self, trainset, model, test_data):

        """ runs a complete eval of the model.
            args: 
                model: your trained rec model.
                test_data: dataframe with cols [user_id, item_id, timestamps]
            returns
                Dict of metrics
        """
        # group test data by user to see what each user interacted with.
        actual_interactions = test_data.groupby('user_id')['item_id'].apply(set).to_dict()
        results = {k: {
            'precision': [],
            'recall': [],
            'f1': [],
            'avg_precision': []
        } for k in self.k_values}

        # evaluate for each user who has test interactions
        for user_id, actual_items in actual_interactions.items():
            try:
                recs = get_top_recommendations(user_id, trainset, model, test_data, n=max(self.k_values))
                recommendations = [item for item, _ in recs] # extracts just items_id
               
            except: # handles coldstart users who weren't in the training data.
                continue

            for k in self.k_values:
                precision = calc_precision_at_k(recommendations, actual_items, k)
                recall = calc_recall_at_k(recommendations, actual_items, k)
                f1 = calc_f1_score(precision, recall)
                ap = calc_average_precision(recommendations, actual_items, k)

                results[k]['precision'].append(precision)
                results[k]['recall'].append(recall)
                results[k]['f1'].append(f1)
                results[k]['avg_precision'].append(ap)

        summary = {}
        for k in self.k_values:
            summary[f'precision@{k}'] = np.mean(results[k]['precision'])
            summary[f'recall@{k}'] = np.mean(results[k]['recall'])
            summary[f'f1@{k}'] = np.mean(results[k]['f1']) 
            summary[f'map@{k}'] = np.mean(results[k]['avg_precision'])

        return summary

def calc_recommendation_diversity(all_recommendations):
    
    # maximum enthropy ensures the recommendations are evenly distrubuted across all itmes
    # counts how many times each items was recommended.

    item_counts = {}
    total_recommendations = 0

    for recommendations in all_recommendations:
        for item in recommendations:
            item_counts[item] = item_counts.get(item, 0) + 1
            total_recommendations += 1

    if len(item_counts) == 0: # handles cases where no recommendations were made.
        return {
            'entropy': 0,
            'normalized_entropy': 0,
            'unique_items_recommend': 0,
            'gini_coefficient': 0
        }

    entropy = 0
    for count in item_counts.values():
        probability = count / total_recommendations
        if probability > 0:
            entropy -= probability * np.log2(probability)

    max_entropy = np.log2(len(item_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'unique_items_recommend': len(item_counts),
        'gini_coefficient': calc_gini(list(item_counts.values()))
    }

def calc_gini(counts):
    """ calculate Gini coefficient to measure inequality in recommendations.
    
    gini = 0 means perfect equality (all items recommended equally)
    gini = 1 means perfect inequality (all recommendations for one item)
    
    for food waste, you want a lower Gini coefficient.
    """
    counts = np.array(sorted(counts))
    n = len(counts)
    index = np.arange(1, n + 1)
    # return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n - 1) / n
    numerator = 2 * np.sum(index * counts)
    denominator = n * np.sum(counts)

    if denominator == 0:
        return 0.0
    
    return (numerator / denominator) - (n + 1) / n

def calc_catalog_coverage(all_recommendations, total_available_items):
    recommended_items = set()
    for recommendations in all_recommendations:
        recommended_items.update(recommendations)

    coverage = len(recommended_items)/len(total_available_items) if len(total_available_items) > 0 else 0.0

    return {
        'coverage': coverage,
        'items_recommended': len(recommended_items),
        'items_never_recommended': len(total_available_items) - len(recommended_items)
    }

# Actionable Reports
def generate_eval_report(model, trainset, train_data, test_data, available_items):
    # generates a comprehensive eval report with interpretations.

    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print()

    metrics = evaluator.evaluate(trainset, model, test_data)

    print("Accuracy Metrics: ")
    print("-" * 60)
    for metric_name, value in sorted(metrics.items()):
        print(f"{metric_name}: {value:.4f}")

        # adding interpolations.
        if 'precision' in metric_name and value < 0.05:
            print(" Low precision - many recommendations aren't relevant.")
        if 'recall' in metric_name and value < 0.10:
            print(" Low recall - many relevant items are being missed.")

    print()

    test_users = test_data['user_id'].unique()
    all_recommendations = []

    for user in test_users:
        try:
            recs = get_top_recommendations(user, trainset, model, train_data, n=20)
            if recs:
                item_ids = [item for item, _ in recs] # stores only the item ids not tuples, diversity & coverage metrics.
                all_recommendations.append(item_ids)
        except Exception as e:
            continue

    
    # calculating diversity metrics
    print("Diversity Metrics: ")
    print("-" * 60)
    diversity_metrics  = calc_recommendation_diversity(all_recommendations)

    print(f"normalized entropy: {diversity_metrics['normalized_entropy']:.4f}")

    if diversity_metrics['normalized_entropy'] < 0.5:
        print("Low diversity - recommendations are too similar or concentrated")
        print("Consider: - Adding more diverse items to the catalog")

    print(f"Gini coefficient: {diversity_metrics['gini_coefficient']:.4f}")

    if diversity_metrics['gini_coefficient'] > 0.7:
        print("High inequality - a few items dominate recommendations")
        print("Consider: - Implmementing a coverage-based  re-ranking")

    print()

    # calc. catalog coverage
    print("Coverage Metrics: ")
    print("-" * 60)
    coverage_metrics = calc_catalog_coverage(all_recommendations, available_items)

    print(f"Catalog coverage: {coverage_metrics['coverage']:.2%}")
    print(f"Items recommended: {coverage_metrics['items_recommended']}")
    print(f"Items never recommended: {coverage_metrics['items_never_recommended']}")

    if coverage_metrics['coverage'] < 0.3:
        print("Low coverage - many items never get recommended")
        print("Consider: Hybrid model with content-based filtering")
        print("Consider: Explore-exploit stragegy to surface new items")
        
    print()
    print("=" * 60)

    return {
        'accuracy': metrics,
        'diversity': diversity_metrics,
        'coverage': coverage_metrics
    }       

# Metrics tracker over time.

class MetricsTracker:
    """Tracks eval metrics over time to monitor model performance."""

    def __init__(self, log_file='metrics_log.json'):
        self.log_file = log_file

    def log_evaluation(self, model_name, model_version, metrics, notes=""):
        
        """
        args: 
            model_name: identifier for the model.
            model_version: version or iteration number.
            metrics: dict of eval metrics.
            notes: any obseervation or changes in this iteration.
        """

        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_version': model_version,
            'metrics': metrics,
            'notes': notes
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_metric_history(self, metric_name):
        # historical values for a specific metric.
        # useful for plotting trends over time.

        history = []

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if metric_name in entry['metrics']:
                        history.append({'timestamp': entry['timestamp'], 
                                        'value': entry['metrics'][metric_name], 
                                        'model_version': entry['model_version'] })
        except FileNotFoundError:
            print(f"No log file found at {self.log_file}")

        return history
    
# Setting Up Continuous Evaluation

def run_compile_eval_pipeline(interactions_df, model_class, model_param):
    # end-to-end eval pipeline you can run at any time.
    """
    This function: 
        1. Splits data temporally
        2. Trains the model
        3. Evaluates the model
        4. Generates a report
        5. Logs the metrics
        6. Provides actionable insights.
    """

    print("Starting evaluation pipeline...")
    print("Splitting data...")
    train_data, val_data, test_data = create_temporal_split(
        interactions_df,
        test_weeks=2,
        validation_weeks=1

    )

    print("\nTraining model...")
    model = model_class(**model_param)
    model.fit(train_data)

    print("\nRunning eval...")
    available_items = set(interactions_df['item_id'].unique())
    report = generate_eval_report(model, trainset, train_data, test_data, available_items)

    print("\nLogging results...")
    tracker = MetricsTracker()
    tracker.log_evaluation(
        model_name=model_class.__name__,
        model_version=datetime.now().strftime("%Y%m%d"),
        metrics={
            **report['accuracy_metrics'],
            **report['diversity_metrics'],
            **report['coverage_metrics']
        },
        notes="baseline eval with temporal splits"
    )

    return model, report

def analyze_cold_start_severity(data):
    user_interactions_counts = data.groupby('user_id').size()
    items_interactions_counts = data.groupby('item_id').size()

    n_users = data['user_id'].nunique()
    n_items = data['item_id'].nunique()

    n_iteractions = len(data)
    possible_interactions = n_users * n_items
    sparsity = 1 - (n_iteractions / possible_interactions)

    print(f"Data Sparsity: {sparsity: .2%}")
    print(f"Cold users (<2 interactions): {(user_interactions_counts <= 2).sum()}")
    print(f"Cold items (<2 interactions): {(items_interactions_counts <= 2).sum()}")

    return sparsity

def build_popularity_baseline(data, top_n=20):
    """
        This is for new users with cold starts, it recommends the most popular items.
    """

    item_popularity = data.groupby('item_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()

    item_popularity.columns = ['item_id', 'interaction_count', 'avg_rating']

    # calc popularity score.
    max_interactions = item_popularity['interaction_count'].max()
    max_rating = item_popularity['avg_rating'].max()

    if max_rating > 0:
        item_popularity['popularity_score'] = (
            0.7 * (item_popularity['interaction_count'] / max_interactions) +
            0.3 * (item_popularity['avg_rating'] / max_rating)
        )
    else:
        item_popularity['popularity_score'] = (
            item_popularity['interaction_count'] / max_interactions
        )

    popularity_sorted = item_popularity.sort_values('popularity_score', ascending=False)

    return popularity_sorted

def get_popularity_recommendations(popularity_df, n=10, exclude_items=None):
    # gets top n recs
    if exclude_items is None:
        exclude_items = set()

    available = popularity_df[~popularity_df['item_id'].isin(exclude_items)]
    top_items = available.head(n)[['item_id', 'popularity_score']].values.tolist()

    return [(item, score) for item, score in top_items]

def get_hybrid_recommendations(user_id, trainset, model, popularity_df, min_interactions=3, n=10):
    """
        handles coldstart users:
            1. warm users: collaborative filtering
            2. lukewarm users: blend collaborative filter & popularity
            3. cold users: popularity-based only
    """

    try:
        user_inner_id = trainset.to_inner_uid(user_id)
        user_interactions = len(trainset.ur[user_inner_id])

        if user_interactions >= min_interactions:
            # WARM USER: use collaborative filtering
            all_items = trainset.all_items()
            user_items = set([trainset.to_raw_iid(item_id)
                for item_id, _ in trainset.ur[user_inner_id]])

            predictions = []
            for item_id in all_items:
                raw_item_id = trainset.to_raw_iid(item_id)
                if raw_item_id not in user_items:
                    pred = model.predict(user_id, raw_item_id)
                    predictions.append((raw_item_id, pred.est))

            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n]

        elif user_interactions > 0:
            # LUKEWARM USER: Blend collaborative and popularity
            all_items = trainset.all_items()
            user_items = set([trainset.to_raw_iid(item_id) 
                            for item_id, _ in trainset.ur[user_inner_id]])

            cf_predictions = []
            for item_id in all_items:
                raw_item_id = trainset.to_raw_iid(item_id)
                if raw_item_id not in user_items:
                    pred = model.predict(user_id, raw_item_id)
                    cf_predictions.append((raw_item_id, pred.est))

            # Normalize CF scores to [0, 1]
            if cf_predictions:
                cf_scores = [score for _, score in cf_predictions]
                min_score, max_score = min(cf_scores), max(cf_scores)
                if max_score > min_score:
                    cf_predictions = [(item, (score - min_score) / (max_score - min_score)) 
                                    for item, score in cf_predictions]

            # Get popularity scores
            pop_dict = dict(zip(popularity_df['item_id'], 
                              popularity_df['popularity_score']))
            
            # Blend based on interaction count
            cf_weight = user_interactions / min_interactions
            pop_weight = 1 - cf_weight
            
            blended = []
            for item, cf_score in cf_predictions:
                pop_score = pop_dict.get(item, 0)
                final_score = cf_weight * cf_score + pop_weight * pop_score
                blended.append((item, final_score))
            
            blended.sort(key=lambda x: x[1], reverse=True)
            return blended[:n]
        
    except ValueError:
        pass  # User not in trainset
    
    # COLD START USER: Use popularity only
    return get_popularity_recommendations(popularity_df, n=n)


def boost_fresh_items(recommendations, item_freshness, freshness_boost=0.15,
                     freshness_days=2):
    """Boost recently added items to give them visibility."""
    boosted = []
    
    for item, score in recommendations:
        days_old = item_freshness.get(item, 999)
        
        if days_old <= freshness_days:
            boost_factor = 1 + freshness_boost * (1 - days_old / freshness_days)
            boosted_score = score * boost_factor
            boosted.append((item, boosted_score))
        else:
            boosted.append((item, score))
    
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def inject_diversity_for_exploration(recommendations, all_items, explore_ratio=0.2):
    """Replace some recommendations with random items for exploration."""
    n_explore = int(len(recommendations) * explore_ratio)
    n_exploit = len(recommendations) - n_explore
    
    final_recs = recommendations[:n_exploit]
    
    recommended_items = {item for item, _ in recommendations}
    unexplored = [item for item in all_items if item not in recommended_items]
    
    if unexplored and n_explore > 0:
        explore_items = np.random.choice(
            unexplored, 
            size=min(n_explore, len(unexplored)), 
            replace=False
        )
        
        if final_recs:
            explore_score = final_recs[-1][1] * 0.9
        else:
            explore_score = 0.5
        
        for item in explore_items:
            final_recs.append((item, explore_score))
    
    return final_recs


def get_production_recommendations(user_id, trainset, model, data, 
                                  available_items=None, n=10,
                                  enable_freshness_boost=True,
                                  enable_exploration=True,
                                  exploration_rate=0.15):
    """
    Production-ready recommendation function that handles all edge cases.
    This is what you'd use in your actual application.
    """
    # Build popularity baseline
    popularity_df = build_popularity_baseline(data)
    
    # Get base recommendations (handles cold start)
    try:
        recommendations = get_hybrid_recommendations(
            user_id, trainset, model, popularity_df,
            min_interactions=5, n=n*3  # Buffer for filtering
        )
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        recommendations = get_popularity_recommendations(popularity_df, n=n*2)
    
    # Apply freshness boost if enabled
    if enable_freshness_boost:
        item_counts = data.groupby('item_id').size()
        item_freshness = {item: max(1, 30 - count) 
                         for item, count in item_counts.items()}
        
        recommendations = boost_fresh_items(
            recommendations, item_freshness,
            freshness_boost=0.15, freshness_days=2
        )
    
    # Filter by availability if provided
    if available_items is not None:
        recommendations = [(item, score) for item, score in recommendations 
                          if item in available_items]
    
    # Take top N before exploration
    recommendations = recommendations[:n]
    
    # Inject exploration if enabled
    if enable_exploration and len(recommendations) > 0:
        all_items = data['item_id'].unique().tolist()
        recommendations = inject_diversity_for_exploration(
            recommendations, all_items, explore_ratio=exploration_rate
        )
    
    return recommendations

# Test Phase 4 at the end of your file
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PHASE 4: COLD START HANDLING")
    print("="*70 + "\n")
    
    # Load data
    data = load_data_correctly()
    
    # Analyze cold start severity
    print("Analyzing cold start severity...")
    sparsity = analyze_cold_start_severity(data)
    print()
    
    # Train model
    print("Training model...")
    reader = Reader(rating_scale=(0, 2))
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    trainset = dataset.build_full_trainset()
    
    model = SVD()
    model.fit(trainset)
    print("✓ Model trained\n")
    
    # Build popularity baseline
    print("Building popularity baseline...")
    popularity_df = build_popularity_baseline(data)
    print(f"✓ Top 5 popular items:")
    for idx, row in popularity_df.head(5).iterrows():
        print(f"  {row['item_id']}: score {row['popularity_score']:.3f}")
    print()
    
    # Test with different user types
    print("Testing recommendations for different user types:")
    print("-"*70)
    
    user_interaction_counts = data.groupby('user_id').size()
    
    # Find a cold user (0 interactions - if any exist in test set)
    # Find a lukewarm user (1-4 interactions)
    # Find a warm user (5+ interactions)
    
    lukewarm_users = user_interaction_counts[
        (user_interaction_counts > 0) & (user_interaction_counts < 5)
    ].index.tolist()
    warm_users = user_interaction_counts[user_interaction_counts >= 5].index.tolist()
    
    test_users = []
    if lukewarm_users:
        test_users.append(('lukewarm', lukewarm_users[0]))
    if warm_users:
        test_users.append(('warm', warm_users[0]))
    
    for user_type, user_id in test_users:
        interaction_count = user_interaction_counts[user_id]
        print(f"\n{user_type.upper()} USER: {user_id} ({interaction_count} interactions)")
        
        # Test hybrid recommendations
        recs = get_hybrid_recommendations(
            user_id, trainset, model, popularity_df,
            min_interactions=5, n=5
        )
        
        print("Recommendations:")
        for item, score in recs:
            print(f"  - {item}: score = {score:.3f}")
    
    # Test production recommendations with all features
    print("\n" + "="*70)
    print("TESTING PRODUCTION RECOMMENDATIONS (with all features)")
    print("="*70)
    
    if test_users:
        test_user = test_users[0][1]
        print(f"\nUser: {test_user}")
        
        prod_recs = get_production_recommendations(
            test_user, trainset, model, data,
            n=5,
            enable_freshness_boost=True,
            enable_exploration=True,
            exploration_rate=0.2
        )
        
        print("Production Recommendations:")
        for item, score in prod_recs:
            print(f"  - {item}: score = {score:.3f}")
    
    print("\n" + "="*70)
    print("✓ PHASE 4 TESTING COMPLETED")
    print("="*70)