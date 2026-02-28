import pandas as pd
from src.types import BooleanCondition
from src.preprocessing import exclude_keys_from_trace
from src.tree_utils import (
    get_positive_paths,
    get_compliant_paths,
    get_missing_conditions
)


def extract_recommendations(tree, feature_names: list, class_values: list, prefix_set: pd.DataFrame) -> dict:
    # Extracts recommendations from a decision tree for traces with negative prediction
   
    recommendations = {}
    
    # Determine positive class from class_values and tree.classes_
    positive_class = class_values[0]   # "true"
    positive_class_idx = list(tree.classes_).index(positive_class)
    
    # Step 1: Extract all positive paths from the tree
    positive_paths = get_positive_paths(tree, feature_names, positive_class_idx)
    print(f"Found {len(positive_paths)} positive paths in the tree\n")
    
    # Step 2: Make predictions internally using the tree
    X = prefix_set[feature_names]
    predictions = tree.predict(X)
    
    # Counters for summary
    n_positive_pred = 0
    n_with_recs = 0
    n_no_recs = 0
    
    # Step 3: For each prefix trace
    for i, (idx, row) in enumerate(prefix_set.iterrows()):
        trace_dict = row.to_dict()
        trace_id = trace_dict.get('trace_id', f'trace_{idx}')
        prediction = predictions[i]
        
        # Extract activities present in the trace (True values only)
        true_activities = exclude_keys_from_trace(trace_dict)
        trace_key = frozenset(true_activities.keys())
        
        # Step 3a: If prediction is positive, return empty set (no recommendations needed)
        if prediction == positive_class:
            recommendations[trace_key] = set()
            n_positive_pred += 1
            continue
        
        # Step 3b: Build current conditions from the trace
        current_conditions = {**true_activities}
        
        # Step 3c: Find compliant positive paths
        compliant_paths = get_compliant_paths(positive_paths, current_conditions)
        
        # If no compatible paths, return empty set
        if not compliant_paths:
            recommendations[trace_key] = set()
            n_no_recs += 1
            continue
        
        # Step 3d: Choose the path with maximum confidence (min length to break ties)
        best_path, best_confidence = max(
            compliant_paths,
            key=lambda item: (item[1], -len(item[0]))
        )
        
        # Step 3e: Extract conditions not already verified in the prefix trace
        missing = get_missing_conditions(best_path, current_conditions)
        recommendations[trace_key] = missing
        
        if missing:
            n_with_recs += 1
        else:
            n_no_recs += 1
    
    print(f"Summary:")
    print(f"  - Traces with positive prediction: {n_positive_pred}")
    print(f"  - Traces with recommendations: {n_with_recs}")
    print(f"  - Negative traces without possible recommendations: {n_no_recs}")
    
    return recommendations


def print_recommendations(recommendations: dict, max_display: int = 5):
    print("RECOMMENDATIONS EXTRACTED")
    
    filtered = {k: v for k, v in recommendations.items() if v is not None and len(v) > 0}
    
    if not filtered:
        print("No recommendations available (all traces are already positive or no recommendations possible)")
        return
    
    count = 0
    for trace_activities, recommendations_set in filtered.items():
        if count >= max_display:
            print(f"\n... and {len(filtered) - max_display} more traces")
            break
        
        count += 1
        print(f"\n[{count}] Trace with activities: {set(trace_activities)}")
        
        if len(recommendations_set) == 0:
            print("No recommendations possible")
        else:
            print("Recommendations:")
            for condition in recommendations_set:
                action = "Add" if condition.value else "Remove"
                print(f"       • {action}: {condition.feature}")


def evaluate_recommendations(test_set: dict, recommendations: dict) -> dict:
    # Evaluate recommendations against complete (non-truncated) test traces.
    
    # Unpack test_set dictionary
    tree = test_set['tree']
    feature_names = test_set['feature_names']
    class_values = test_set['class_values']
    prefix_test_set = test_set['prefix_test_set']
    full_test_set = test_set['full_test_set']
    
    positive_class = class_values[0]
    
    # Predict on prefix test set
    X = prefix_test_set[feature_names]
    predictions = tree.predict(X)
    
    tp = tn = fp = fn = 0
    evaluated = 0
    
    for i in range(len(prefix_test_set)):
        prediction = predictions[i]
        
        # Skip positive predictions (no recommendation needed)
        if prediction == positive_class:
            continue
        
        evaluated += 1
        
        # Get recommendation key from prefix trace
        prefix_row = prefix_test_set.iloc[i]
        prefix_dict = prefix_row.to_dict()
        true_activities = exclude_keys_from_trace(prefix_dict)
        key = frozenset(true_activities.keys())
        
        # Look up recommendation
        recs = recommendations.get(key, set())
        
        # Get full trace activities
        full_row = full_test_set.iloc[i]
        full_dict = full_row.to_dict()
        full_activities = exclude_keys_from_trace(full_dict)
        full_features = frozenset(full_activities.keys())
        
        # Check if recommendations are followed in the full trace
        recommendation_followed = True
        
        if not recs or len(recs) == 0:
            # No recommendation could be made -> not followed
            recommendation_followed = False
        else:
            for condition in recs:
                should_be_present = condition.value
                is_present = condition.feature in full_features
                
                if should_be_present != is_present:
                    recommendation_followed = False
                    break
        
        # Ground truth from full trace
        ground_truth = full_row['label']
        
        # Classify
        if recommendation_followed and ground_truth == 'true':
            tp += 1
        elif not recommendation_followed and ground_truth == 'false':
            tn += 1
        elif recommendation_followed and ground_truth == 'false':
            fp += 1
        elif not recommendation_followed and ground_truth == 'true':
            fn += 1
    
    total = tp + tn + fp + fn
    
    if total == 0:
        return {
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'accuracy': 0.0,
            'evaluated': 0
        }
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / total
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(acc, 4),
        'evaluated': evaluated
    }
    
