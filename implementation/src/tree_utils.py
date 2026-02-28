import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from src.types import BooleanCondition, ThresholdCondition, Condition, Path


def is_leaf(tree_, node_id: int) -> bool:
    # Verifies if a node is a leaf 
    return tree_.feature[node_id] == _tree.TREE_UNDEFINED


def compute_path_confidence(tree_, node_id: int, positive_class_idx: int = 1) -> float:
    # Computes the confidence (class probability) of a positive leaf

    # tree_.value[node_id][0] = [n_class_0, n_class_1, ...]
    node_values = tree_.value[node_id][0]
    n_positive = node_values[positive_class_idx]
    
    node_samples = tree_.n_node_samples[node_id]
    
    return n_positive / node_samples


def get_positive_paths(tree: DecisionTreeClassifier, feature_names: list, positive_class_idx: int = 1) -> list[tuple[Path, float]]:
    # Extracts all paths from root to positive leaves using DFS
  
    tree_ = tree.tree_
    paths = []
    
    def dfs_traverse(node_id, current_path):
        # Recursive DFS traversal of the tree

        # Base case: leaf node
        if is_leaf(tree_, node_id):
            predicted_class_idx = np.argmax(tree_.value[node_id][0])
            
            if predicted_class_idx == positive_class_idx:
                confidence = compute_path_confidence(tree_, node_id, positive_class_idx)
                paths.append((current_path.copy(), confidence))
            return
        
        # Recursive case: internal node
        feature_idx = tree_.feature[node_id]
        feature_name = feature_names[feature_idx]
        threshold = tree_.threshold[node_id]
        

        if feature_name == 'prefix_length':
            # Left child: value <= threshold
            left_condition = ThresholdCondition(
                feature=feature_name,
                op='<=',
                threshold=threshold
            )
            dfs_traverse(tree_.children_left[node_id], current_path + [left_condition])
            
            # Right child: value > threshold
            right_condition = ThresholdCondition(
                feature=feature_name,
                op='>',
                threshold=threshold
            )
            dfs_traverse(tree_.children_right[node_id], current_path + [right_condition])
        else:
            left_condition = BooleanCondition(feature=feature_name, value=False)
            dfs_traverse(tree_.children_left[node_id], current_path + [left_condition])
            
            right_condition = BooleanCondition(feature=feature_name, value=True)
            dfs_traverse(tree_.children_right[node_id], current_path + [right_condition])
    
    # Start DFS from the root (node 0)
    dfs_traverse(0, [])
    
    return paths


def contradicts(condition: Condition, trace_dict: dict) -> bool:
    # Checks if a condition contradicts the current trace state
    if isinstance(condition, BooleanCondition):
        # Feature not present in trace means no contradiction
        if condition.feature not in trace_dict:
            return False
        return trace_dict[condition.feature] != condition.value
    
    elif isinstance(condition, ThresholdCondition):
        # Threshold condition (e.g. prefix_length)
        if condition.feature not in trace_dict:
            return False
        
        value = trace_dict[condition.feature]
        if condition.op == '<=':
            return value > condition.threshold
        elif condition.op == '>':
            return value <= condition.threshold
        else:
            raise ValueError(f"Unknown operator: {condition.op}")
    
    return False


def get_compliant_paths(paths: list[tuple[Path, float]], current_conditions: dict) -> list[tuple[Path, float]]:
    # Filters paths that are compliant with the current trace conditions

    compliant = []
    
    for path, confidence in paths:
        is_compliant = True
        
        for condition in path:
            if contradicts(condition, current_conditions):
                is_compliant = False
                break
        
        if is_compliant:
            compliant.append((path, confidence))
    
    return compliant


def get_missing_conditions(path: Path, current_conditions: dict) -> set[BooleanCondition]:
    # Extracts conditions from the path not already satisfied in the prefix trace

    recommendations = []
    
    for condition in path:
        if isinstance(condition, BooleanCondition):
            # If feature is absent from current_conditions, default is False
            actual_value = current_conditions.get(condition.feature, False)
            if actual_value != condition.value:
                recommendations.append(condition)
    
    return set(recommendations)
