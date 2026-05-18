# ================================================================
#  Харилцагчийн Churn Таамаглах — Decision Tree ӨӨРӨӨ ХЭРЭГЖҮҮЛСЭН
#  sklearn.tree, sklearn.ensemble ашиглахгүй!
#  Зөвхөн: numpy, pandas, matplotlib, seaborn, collections
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("=" * 60)
print("  Decision Tree — sklearn АШИГЛАХГҮЙ хэрэгжүүлэлт")
print("=" * 60)


# ================================================================
#  ХЭСЭГ 1: ҮНДСЭН ФУНКЦҮҮД
# ================================================================

def gini_impurity(y):
    """Gini = 1 - sum(pi^2)"""
    n = len(y)
    if n == 0:
        return 0.0
    impurity = 1.0
    for count in Counter(y).values():
        p = count / n
        impurity -= p ** 2
    return impurity


def weighted_gini(y_left, y_right):
    """Жинлэгдсэн Gini хоёр хүүхэд node-д"""
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left)/n)*gini_impurity(y_left) + \
           (len(y_right)/n)*gini_impurity(y_right)


def best_split(X, y):
    """Хамгийн бага Weighted Gini өгдөг (feature, threshold) олох"""
    best_gini, best_feature, best_thresh = float('inf'), None, None
    for feat_idx in range(X.shape[1]):
        col = X[:, feat_idx]
        for thresh in sorted(set(col))[:-1]:
            mask_l = col <= thresh
            mask_r = col > thresh
            if mask_l.sum() == 0 or mask_r.sum() == 0:
                continue
            g = weighted_gini(y[mask_l], y[mask_r])
            if g < best_gini:
                best_gini, best_feature, best_thresh = g, feat_idx, thresh
    return best_feature, best_thresh, best_gini


# ================================================================
#  ХЭСЭГ 2: NODE КЛАСС
# ================================================================

class Node:
    def __init__(self, feature_idx=None, threshold=None,
                 left=None, right=None, value=None,
                 gini=0.0, n_samples=0, depth=0):
        self.feature_idx = feature_idx
        self.threshold   = threshold
        self.left        = left
        self.right       = right
        self.value       = value
        self.gini        = gini
        self.n_samples   = n_samples
        self.depth       = depth

    def is_leaf(self):
        return self.value is not None


# ================================================================
#  ХЭСЭГ 3: DECISION TREE CLASSIFIER
# ================================================================

class DecisionTreeClassifier:
    """
    Decision Tree ангилагч — sklearn ашиглахгүй, бүрэн өөрийн хэрэгжүүлэлт.
    Алгоритм: Gini Impurity-д суурилсан CART
    """
    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.root               = None
        self.feature_importances_ = None
        self._n_features        = 0
        self._importance_raw    = None
        self._n_total           = 0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self._n_features     = X.shape[1]
        self._n_total        = len(y)
        self._importance_raw = np.zeros(self._n_features)
        self.root = self._build(X, y, 0)
        total = self._importance_raw.sum()
        self.feature_importances_ = (
            self._importance_raw / total if total > 0
            else self._importance_raw
        )
        return self

    def _build(self, X, y, depth):
        n         = len(y)
        node_gini = gini_impurity(y)

        # Зогсоох нөхцөл
        if depth >= self.max_depth or n < self.min_samples_split or node_gini == 0.0:
            return Node(value=self._majority(y), gini=node_gini, n_samples=n, depth=depth)

        feat_idx, thresh, split_gini = best_split(X, y)

        if feat_idx is None:
            return Node(value=self._majority(y), gini=node_gini, n_samples=n, depth=depth)

        # Feature importance шинэчлэх
        self._importance_raw[feat_idx] += (node_gini - split_gini) * (n / self._n_total)

        mask   = X[:, feat_idx] <= thresh
        X_l, y_l = X[mask],  y[mask]
        X_r, y_r = X[~mask], y[~mask]

        if len(y_l) < self.min_samples_leaf or len(y_r) < self.min_samples_leaf:
            return Node(value=self._majority(y), gini=node_gini, n_samples=n, depth=depth)

        return Node(
            feature_idx=feat_idx, threshold=thresh,
            left=self._build(X_l, y_l, depth+1),
            right=self._build(X_r, y_r, depth+1),
            gini=node_gini, n_samples=n, depth=depth
        )

    def _majority(self, y):
        return max(Counter(y), key=Counter(y).get)

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._walk(x, self.root) for x in X])

    def _walk(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._walk(x, node.left)
        return self._walk(x, node.right)

    def print_tree(self, node=None, names=None, indent="", max_d=3, cur_d=0):
        if node is None:
            node = self.root
        if cur_d > max_d:
            return
        if node.is_leaf():
            lbl = "Гарна" if node.value == 1 else "Үлдэнэ"
            print(f"{indent}НАВЧ -> {lbl}  (n={node.n_samples}, gini={node.gini:.3f})")
            return
        fname = names[node.feature_idx] if names else f"f[{node.feature_idx}]"
        print(f"{indent}[{fname} <= {node.threshold:.3f}]  gini={node.gini:.3f}  n={node.n_samples}")
        print(f"{indent}|-- Tiim:")
        self.print_tree(node.left,  names, indent+"|   ", max_d, cur_d+1)
        print(f"{indent}+-- Ugui:")
        self.print_tree(node.right, names, indent+"    ", max_d, cur_d+1)

    def visualize(self, names=None, max_depth=3, save='my_tree_viz.png'):
        fig, ax = plt.subplots(figsize=(22, 11))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

        def draw(node, x, y, dx, d):
            if node is None or d > max_depth:
                return
            if node.is_leaf():
                color = '#d5f5e3' if node.value == 0 else '#fadbd8'
                edge  = '#27ae60' if node.value == 0 else '#c0392b'
                lbl   = "Uldene" if node.value == 0 else "Garna"
                txt   = f"{lbl}\nn={node.n_samples}\ngini={node.gini:.3f}"
            else:
                color, edge = '#d6eaf8', '#2980b9'
                fn  = names[node.feature_idx] if names else f"f{node.feature_idx}"
                txt = f"{fn}\n<={node.threshold:.2f}\nn={node.n_samples}"
            ax.text(x, y, txt, ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                              edgecolor=edge, linewidth=1.2))
            if not node.is_leaf() and d < max_depth:
                cy  = y - 0.17
                ndx = dx / 2
                for child, xc, lbl, clr in [
                        (node.left,  x-dx, 'Yes', '#27ae60'),
                        (node.right, x+dx, 'No',  '#c0392b')]:
                    ax.annotate("", xy=(xc, cy+0.04), xytext=(x, y-0.055),
                                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
                    ax.text((x+xc)/2, (y+cy)/2+0.01, lbl,
                            fontsize=7.5, color=clr, ha='center', fontweight='bold')
                    draw(child, xc, cy, ndx, d+1)

        draw(self.root, 0.5, 0.92, 0.22, 0)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#d6eaf8', edgecolor='#2980b9', label='Internal node'),
            Patch(facecolor='#d5f5e3', edgecolor='#27ae60', label='Leaf: Uldene (0)'),
            Patch(facecolor='#fadbd8', edgecolor='#c0392b', label='Leaf: Garna (1)'),
        ], loc='lower right', fontsize=9)
        ax.set_title('Churn Decision Tree (No sklearn)', fontsize=12)
        plt.tight_layout()
        plt.savefig(save, dpi=160, bbox_inches='tight')
        plt.show()
        print(f"  [OK] {save} saved")


# ================================================================
#  ХЭСЭГ 4: МЕТРИКҮҮД 
# ================================================================

def evaluate(y_true, y_pred, label=""):
    tp = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==1)
    tn = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==0)
    fp = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==1)
    fn = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==0)
    n    = tp+tn+fp+fn
    acc  = (tp+tn)/n            if n>0         else 0
    prec = tp/(tp+fp)           if (tp+fp)>0   else 0
    rec  = tp/(tp+fn)           if (tp+fn)>0   else 0
    f1   = 2*prec*rec/(prec+rec)if (prec+rec)>0 else 0
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    return acc, prec, rec, f1


def plot_cm(y_true, y_pred):
    tp = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==1)
    tn = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==0)
    fp = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==1)
    fn = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==0)
    cm = np.array([[tn, fp],[fn, tp]])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Uldene','Garna'],
                yticklabels=['Uldene','Garna'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Bodit')
    ax.set_xlabel('Tamaglasen')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("  [OK] confusion_matrix.png saved")


def plot_importance(importances, names):
    feat_df = pd.DataFrame({'Feature': names, 'Importance': importances}) \
                .sort_values('Importance', ascending=False).reset_index(drop=True)
    print("\nFeature Importance:")
    print(feat_df.to_string(index=False))
    colors = ['#1d9e75' if i < 3 else '#aaa' for i in range(len(feat_df))]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
    ax.invert_yaxis()
    ax.set_title('Feature Importance (Gini-based)')
    ax.set_xlabel('Importance score')
    for i, v in enumerate(feat_df['Importance']):
        ax.text(v+0.002, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    print("  [OK] feature_importance.png saved")
    return feat_df


# ================================================================
#  ХЭСЭГ 5: ӨГӨГДӨЛ АЧААЛАХ & БОЛОВСРУУЛАХ
# ================================================================

print("\n[1] Loading data...")
df = pd.read_csv("Churn_Modelling.csv")
print(f"    Shape: {df.shape}")
print(f"    Churn rate: {df['Exited'].mean()*100:.1f}%")

print("\n[2] Preprocessing...")
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)
print(f"    Columns: {list(df.columns)}")

X = df.drop('Exited', axis=1).values.astype(float)
y = df['Exited'].values.astype(int)
feature_names = list(df.drop('Exited', axis=1).columns)


#  ХЭСЭГ 6: TRAIN/TEST ХУВААХ (sklearn ашиглахгүй)


print("\n[3] Train/Test split (80/20)...")
np.random.seed(42)
idx   = np.random.permutation(len(y))
split = int(len(y) * 0.8)
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]
print(f"    Train: {X_train.shape}  churn={y_train.mean()*100:.1f}%")
print(f"    Test : {X_test.shape}  churn={y_test.mean()*100:.1f}%")


#  ХЭСЭГ 7: МОДЕЛ СУРГАХ

print("\n[4] Training Decision Tree...")
print("    (Testing all features and thresholds — may take a few minutes)")

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10)
clf.fit(X_train, y_train)
print("    [OK] Training complete!")


#  ХЭСЭГ 8: ТААМАГЛАЛ & ҮНЭЛГЭЭ


print("\n[5] Predicting...")
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

evaluate(y_train, y_pred_train, "TRAIN evaluation")
evaluate(y_test,  y_pred_test,  "TEST evaluation")

train_acc = sum(y_train == y_pred_train) / len(y_train)
test_acc  = sum(y_test  == y_pred_test)  / len(y_test)
diff = train_acc - test_acc
print(f"\n  Overfitting gap: {diff:.4f}  {'[WARN] Overfit!' if diff > 0.05 else '[OK] Normal'}")

#  ХЭСЭГ 9: МОДНЫ БҮТЭЦ ХЭВЛЭХ

print("\n[6] Tree structure (depth 3):")
print("─" * 60)
clf.print_tree(names=feature_names, max_d=3)


#  ХЭСЭГ 10: ГРАФИКУУД


print("\n[7] Generating plots...")
plot_cm(y_test, y_pred_test)
feat_df = plot_importance(clf.feature_importances_, feature_names)
clf.visualize(names=feature_names, max_depth=3, save='my_tree_viz.png')


#  ХЭСЭГ 11: ШИНЭ ХАРИЛЦАГЧ ТААМАГЛАХ


print("\n[8] Predicting new customers:")
print("─" * 60)
new_customers = np.array([
    [620, 1, 45, 3, 120000, 1, 1, 0, 80000,  1, 0],
    [750, 0, 32, 8, 0,      2, 1, 1, 120000, 0, 0],
    [580, 0, 50, 2, 85000,  1, 0, 0, 60000,  1, 0],
], dtype=float)
descs = ["High risk", "Stable", "Medium risk"]

for i, (row, desc) in enumerate(zip(new_customers, descs)):
    pred = clf._walk(row, clf.root)
    res  = "GARNA (1) [CHURN]" if pred == 1 else "ULDENE (0) [STAY]"
    print(f"  Customer {i+1} ({desc}): {res}")


#  ЭЦСИЙН ТАЙЛАН


print("\n" + "=" * 60)
print("  [DONE] Saved files:")
print("     - confusion_matrix.png")
print("     - feature_importance.png")
print("     - my_tree_viz.png")
print(f"  Top feature: '{feat_df.iloc[0]['Feature']}'")
print(f"  No sklearn.tree used — full custom implementation")
print("=" * 60)
