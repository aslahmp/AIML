import pandas as pd
import numpy as np

file_path = r'C:\Users\anuar\OneDrive\Desktop\dtree_data.xlsx'
df = pd.read_excel(file_path)

print("Dataset preview:")
print(df)

df["AGE"] = df["AGE"].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
df["AGE"] = df["AGE"].str.replace('30', 'a')
df["AGE"] = df["AGE"].str.replace('3140', 'b')
df["AGE"] = df["AGE"].str.replace('40', 'c')

df["INCOME"] = df["INCOME"].str.replace('high', 'a')
df["INCOME"] = df["INCOME"].str.replace('medium', 'b')
df["INCOME"] = df["INCOME"].str.replace('low', 'c')

df["STUDENT"] = df["STUDENT"].str.replace('no', 'a')
df["STUDENT"] = df["STUDENT"].str.replace('yes', 'b')

df["CREDIT"] = df["CREDIT"].str.replace('fair', 'a')
df["CREDIT"] = df["CREDIT"].str.replace('excellent', 'b')

df["BUYS COMPUTER"] = df["BUYS COMPUTER"].str.replace('no', '0')
df["BUYS COMPUTER"] = df["BUYS COMPUTER"].str.replace('yes', '1')

print("Processed dataset:")
print(df)

pos = (df['BUYS COMPUTER'] == '1').sum()
neg = (df['BUYS COMPUTER'] == '0').sum()

print(f"\nPositive labels: {pos}, Negative labels: {neg}")

total_count = pos + neg
print(f"\nTotal Count={total_count}")

parent_entropy = (-(pos / total_count) * np.log2(pos / total_count)
                  - (neg / total_count) * np.log2(neg / total_count))
print(f"\nParent entropy={parent_entropy}")

def calculate_entropy(feature, labels):
    pos_count = {'a': 0, 'b': 0, 'c': 0}
    neg_count = {'a': 0, 'b': 0, 'c': 0}

    pos_feature = feature[labels == '1']
    neg_feature = feature[labels == '0']

    for m in pos_feature:
        words = str(m).split()
        for word in words:
            if word in pos_count:
                pos_count[word] += 1

    for m in neg_feature:
        words = str(m).split()
        for word in words:
            if word in neg_count:
                neg_count[word] += 1

    p_a = pos_count['a'] + neg_count['a']
    p_b = pos_count['b'] + neg_count['b']
    p_c = pos_count['c'] + neg_count['c']

    p_a_1 = (pos_count['a'] / p_a) if p_a > 0 else 0
    p_a_2 = (neg_count['a'] / p_a) if p_a > 0 else 0
    p_b_1 = (pos_count['b'] / p_b) if p_b > 0 else 0
    p_b_2 = (neg_count['b'] / p_b) if p_b > 0 else 0
    p_c_1 = (pos_count['c'] / p_c) if p_c > 0 else 0
    p_c_2 = (neg_count['c'] / p_c) if p_c > 0 else 0

    p_a_1 = p_a_1 if p_a_1 > 0 else 1
    p_a_2 = p_a_2 if p_a_2 > 0 else 1
    p_b_1 = p_b_1 if p_b_1 > 0 else 1
    p_b_2 = p_b_2 if p_b_2 > 0 else 1
    p_c_1 = p_c_1 if p_c_1 > 0 else 1
    p_c_2 = p_c_2 if p_c_2 > 0 else 1

    entropy_feature = (
        (p_a / total_count) * (-p_a_1 * np.log2(p_a_1) - p_a_2 * np.log2(p_a_2)) +
        (p_b / total_count) * (-p_b_1 * np.log2(p_b_1) - p_b_2 * np.log2(p_b_2)) +
        (p_c / total_count) * (-p_c_1 * np.log2(p_c_1) - p_c_2 * np.log2(p_c_2))
    )

    return entropy_feature

features = ['AGE', 'INCOME', 'STUDENT', 'CREDIT']
max_gain_feature = None
max_gain_value = 0

for feature in features:
    entropy_feature = calculate_entropy(df[feature], df['BUYS COMPUTER'])
    gain_feature = parent_entropy - entropy_feature

    if gain_feature > max_gain_value:
        max_gain_value = gain_feature
        max_gain_feature = feature

def split_dataset(df, feature):
    splits = {}

    for value in df[feature].unique():
        splits[value] = df[df[feature] == value]

    return splits

splits = split_dataset(df, max_gain_feature)

def build_tree(df, features, target):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    if len(features) == 0:
        return df[target].mode()[0]

    # Calculate parent entropy
    pos = (df[target] == '1').sum()
    neg = (df[target] == '0').sum()

    total_count = pos + neg

    total_count = total_count if total_count > 0 else 1
    pos = pos if pos > 0 else 1
    neg = neg if neg > 0 else 1

    parent_entropy = (-(pos / total_count) * np.log2(pos / total_count)
                      - (neg / total_count) * np.log2(neg / total_count)) if total_count > 0 else 0

    max_gain_feature = None
    max_gain_value = -np.inf

    for feature in features:
        entropy_feature = calculate_entropy(df[feature], df[target])
        gain_feature = parent_entropy - entropy_feature

        print(f"Calculating gain for feature '{feature}':")
        print(f"Entropy: {entropy_feature}, Gain: {gain_feature}")

        if gain_feature > max_gain_value:
            max_gain_value = gain_feature
            max_gain_feature = feature

    print(f"\nSelected feature '{max_gain_feature}' with maximum gain of {max_gain_value}\n")

    tree_node = {max_gain_feature: {}}

    splits = split_dataset(df, max_gain_feature)

    remaining_features = [f for f in features if f != max_gain_feature]

    for value, subset in splits.items():
        print(f"Building subtree for '{max_gain_feature}={value}' subset:")

        pos_subset = (subset[target] == '1').sum()
        neg_subset = (subset[target] == '0').sum()

        total_subset_count = pos_subset + neg_subset

        total_subset_count = total_subset_count if total_subset_count > 0 else 1
        pos_subset = pos_subset if pos_subset > 0 else 1
        neg_subset = neg_subset if neg_subset > 0 else 1

        subset_entropy = (-(pos_subset / total_subset_count) * np.log2(pos_subset / total_subset_count)
                          - (neg_subset / total_subset_count) * np.log2(
                    neg_subset / total_subset_count)) if total_subset_count > 0 else 0

        print(f"Subset size: {total_subset_count}, Positive labels: {pos_subset}, Negative labels: {neg_subset}")
        print(f"Subset entropy: {subset_entropy}")

        subtree_result = build_tree(subset, remaining_features, target)
        tree_node[max_gain_feature][value] = subtree_result

    return tree_node

features_list = ['AGE', 'INCOME', 'STUDENT', 'CREDIT']
decision_tree = build_tree(df, features_list, 'BUYS COMPUTER')

print("\nFinal Decision Tree:")
print(decision_tree)

from anytree import Node, RenderTree

data =decision_tree

def create_tree(data, parent=None):
    for key, value in data.items():
        node = Node(key, parent=parent)
        if isinstance(value, dict):
            create_tree(value, parent=node)
        else:
            Node(value, parent=node)

root = Node("\nRoot")
create_tree(data, parent=root)

for pre, _, node in RenderTree(root):
    print(f"{pre}{node.name}")



