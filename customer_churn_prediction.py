# ================================================================
#  Харилцагчийн Сэтгэл Ханамж Таамаглах — Decision Tree
#  Kaggle: Bank Customer Churn Prediction (Churn_Modelling.csv)
#  Суулгах: pip install pandas numpy matplotlib seaborn scikit-learn dtreeviz
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier, plot_tree,
    export_text, export_graphviz
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)


# ---------------------------------------------------------------
# АЛХАМ 1: Өгөгдөл ачаалах
# Татах: kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
# ---------------------------------------------------------------
df = pd.read_csv("Churn_Modelling.csv")

print("=" * 55)
print("АЛХАМ 1: Өгөгдлийн ерөнхий мэдээлэл")
print("=" * 55)
print(f"Хэмжээ     : {df.shape}")
print(f"Баганууд   : {list(df.columns)}")
print(f"\nХоосон утга:\n{df.isnull().sum()}")
print(f"\nZорилтот хувьсагч (Exited):\n{df['Exited'].value_counts()}")
print(f"\nChurn хувь : {df['Exited'].mean()*100:.1f}%")


# ---------------------------------------------------------------
# АЛХАМ 2: Өгөгдлийн шинжилгээ (EDA)
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("АЛХАМ 2: Өгөгдлийн шинжилгээ (EDA)")
print("=" * 55)
print(df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']].describe())

# --- График 1: Үндсэн хувьсагчдын харилцаа ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Насны тархалт
df.groupby('Exited')['Age'].plot(kind='kde', ax=axes[0,0], legend=True)
axes[0,0].set_title('Насны тархалт (0=Үлдэнэ, 1=Гарна)')
axes[0,0].set_xlabel('Нас')

# Бүтээгдэхүүний тоо
sns.countplot(data=df, x='NumOfProducts', hue='Exited', ax=axes[0,1],
              palette=['#2ecc71','#e74c3c'])
axes[0,1].set_title('Бүтээгдэхүүний тоо vs Churn')

# Идэвхтэй гишүүн
sns.countplot(data=df, x='IsActiveMember', hue='Exited', ax=axes[0,2],
              palette=['#2ecc71','#e74c3c'])
axes[0,2].set_title('Идэвхтэй гишүүн vs Churn')
axes[0,2].set_xticklabels(['Идэвхгүй', 'Идэвхтэй'])

# Улс vs Churn
sns.countplot(data=df, x='Geography', hue='Exited', ax=axes[1,0],
              palette=['#2ecc71','#e74c3c'])
axes[1,0].set_title('Улс vs Churn')

# Дансны үлдэгдэл
df[df['Exited']==0]['Balance'].plot(kind='hist', ax=axes[1,1],
    alpha=0.6, label='Үлдэнэ', color='#2ecc71', bins=30)
df[df['Exited']==1]['Balance'].plot(kind='hist', ax=axes[1,1],
    alpha=0.6, label='Гарна', color='#e74c3c', bins=30)
axes[1,1].set_title('Дансны үлдэгдэл vs Churn')
axes[1,1].legend()

# Зорилтот хувьсагч тархалт
df['Exited'].value_counts().plot(kind='pie', ax=axes[1,2],
    labels=['Үлдэнэ (80%)', 'Гарна (20%)'],
    colors=['#2ecc71','#e74c3c'], autopct='%1.1f%%', startangle=90)
axes[1,2].set_title('Churn тархалт')
axes[1,2].set_ylabel('')

plt.suptitle('Харилцагчийн Churn — Өгөгдлийн шинжилгээ (EDA)', fontsize=14)
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("eda_plots.png хадгалагдлаа")

# --- График 2: Корреляцийн дулааны газрын зураг ---
plt.figure(figsize=(11, 7))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Хувьсагчдын хоорондын корреляци', fontsize=13)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("correlation_heatmap.png хадгалагдлаа")


# ---------------------------------------------------------------
# АЛХАМ 3: Өгөгдөл цэвэрлэх & Feature Engineering
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("АЛХАМ 3: Өгөгдөл цэвэрлэх & Encoding")
print("=" * 55)

# Хэрэгцээгүй баганыг хасах
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# Gender: Male=1, Female=0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Geography: One-Hot Encoding
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
# → Geography_Germany, Geography_Spain үүснэ (France суурь болно)

print(f"Encoding-ийн дараах баганууд:\n{list(df.columns)}")
print(f"\nДата хэмжээ: {df.shape}")

# Feature ба зорилтот хувьсагч тусгаарлах
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train/Test хуваах — stratify=y нь тэнцвэрийг хадгалана
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # 80/20 churn харьцааг хоёуланд тэгш хуваарилна
)

print(f"\nTrain хэмжээ : {X_train.shape}")
print(f"Test  хэмжээ : {X_test.shape}")
print(f"Train churn  : {y_train.mean()*100:.1f}%")
print(f"Test  churn  : {y_test.mean()*100:.1f}%")


# ---------------------------------------------------------------
# АЛХАМ 4: Decision Tree загвар сургах
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("АЛХАМ 4: Decision Tree загвар")
print("=" * 55)

dt = DecisionTreeClassifier(
    max_depth=5,              # Гүнийг хязгаарлах — overfitting эсэргүүцнэ
    min_samples_split=20,     # Салаалахын тулд хамгийн бага дээж
    min_samples_leaf=10,      # Навч бүрт хамгийн бага дээж
    criterion='gini',         # Gini impurity — хуваалтын чанар
    class_weight='balanced',  # Тэнцвэргүй байдлыг автоматаар засах
    random_state=42
)
dt.fit(X_train, y_train)

# Таамаглал
y_pred  = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)[:, 1]  # Churn магадлал

print(f"Train нарийвчлал : {dt.score(X_train, y_train):.4f}")
print(f"Test  нарийвчлал : {dt.score(X_test,  y_test):.4f}")

# Overfitting шалгах
train_acc = dt.score(X_train, y_train)
test_acc  = dt.score(X_test,  y_test)
diff = train_acc - test_acc
print(f"\nOverfitting зөрүү: {diff:.4f} {'⚠️ Анхаар!' if diff > 0.05 else '✅ Хэвийн'}")


# ---------------------------------------------------------------
# АЛХАМ 5: Decision Tree визуализаци — 3 арга
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("АЛХАМ 5: Decision Tree визуализаци")
print("=" * 55)

# --- АРГА 1: plot_tree (sklearn стандарт, өнгөтэй) ---
plt.figure(figsize=(26, 11))
plot_tree(
    dt,
    feature_names = list(X.columns),
    class_names   = ['Үлдэнэ', 'Гарна'],
    max_depth     = 3,          # Зөвхөн дээд 3 гүн харуулах
    filled        = True,       # Анги бүрийн өнгө
    rounded       = True,       # Дугуй булан
    fontsize      = 9,
    impurity      = True,       # Gini харуулах
    proportion    = True        # Харьцаагаар харуулах
)
plt.title('Харилцагчийн Churn — Decision Tree (гүн 3)', fontsize=14)
plt.savefig('tree_plot_tree.png', dpi=200, bbox_inches='tight')
plt.show()
print("tree_plot_tree.png хадгалагдлаа")

# --- АРГА 2: export_text (тайлан дотор оруулах текст хэлбэр) ---
print("\n--- Модны бүтэц (текст, гүн 3) ---")
tree_rules = export_text(
    dt,
    feature_names = list(X.columns),
    max_depth     = 3
)
print(tree_rules)

# Текст файлд хадгалах
with open('tree_rules.txt', 'w', encoding='utf-8') as f:
    f.write("Decision Tree Rules (depth=3)\n")
    f.write("=" * 40 + "\n")
    f.write(tree_rules)
print("tree_rules.txt хадгалагдлаа")

# --- АРГА 3: export_graphviz → DOT файл (Graphviz суусан бол PDF болгоно) ---
export_graphviz(
    dt,
    out_file      = 'tree_graphviz.dot',
    feature_names = list(X.columns),
    class_names   = ['Үлдэнэ', 'Гарна'],
    max_depth     = 4,
    filled        = True,
    rounded       = True,
    special_characters = True
)
print("tree_graphviz.dot хадгалагдлаа")
print("  → Graphviz суусан бол: dot -Tpng tree_graphviz.dot -o tree.png")

# --- АРГА 4: dtreeviz (хамгийн үзэсгэлэнтэй — нэмэлт суулгалт шаардана) ---
try:
    import dtreeviz
    viz = dtreeviz.model(
        dt,
        X_train       = X_train,
        y_train       = y_train,
        feature_names = list(X.columns),
        class_names   = ['Үлдэнэ', 'Гарна'],
        target_name   = 'Exited'
    )
    v = viz.view(depth_range_to_display=(0, 3))
    v.save('tree_dtreeviz.svg')
    print("tree_dtreeviz.svg хадгалагдлаа")
except ImportError:
    print("dtreeviz суугаагүй байна: pip install dtreeviz")


# ---------------------------------------------------------------
# АЛХАМ 6: Feature Importance — хувьсагчдын үүрэг
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("АЛХАМ 6: Feature Importance")
print("=" * 55)

feat_df = pd.DataFrame({
    'Feature':    X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(feat_df.to_string(index=False))

# График
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#1d9e75' if i < 3 else '#aaaaaa' for i in range(len(feat_df))]
ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
ax.invert_yaxis()
ax.set_title('Хувьсагч бүрийн нөлөө — Gini Importance', fontsize=13)
ax.set_xlabel('Чухлын зэрэг')
for i, (val, name) in enumerate(zip(feat_df['Importance'], feat_df['Feature'])):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("feature_importance.png хадгалагдлаа")

# Дүгнэлтэд бичих мэдээлэл
print("\nТайлан дээр ашиглах дүгнэлт:")
for i, row in feat_df.head(3).iterrows():
    print(f"  {i+1}. '{row['Feature']}' — нийт нөлөөний {row['Importance']*100:.1f}%")


# ---------------------------------------------------------------
# АЛХАМ 7: Загварын үнэлгээ
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("АЛХАМ 7: Загварын үнэлгээ")
print("=" * 55)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}  ← Churn гэж таамагласны хэд нь зөв?")
print(f"Recall    : {rec:.4f}  ← Бодит Churn-ийн хэдийг олсон?")
print(f"F1 Score  : {f1:.4f}  ← Precision & Recall-ын тэнцвэр")
print(f"ROC-AUC   : {auc:.4f}  ← 1.0 = төгс, 0.5 = санамсаргүй")

print("\nДэлгэрэнгүй тайлан:")
print(classification_report(
    y_test, y_pred,
    target_names=['Үлдэнэ (0)', 'Гарна (1)']
))

# --- Confusion Matrix + ROC Curve ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Үлдэнэ', 'Гарна'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Confusion Matrix', fontsize=12)

# Confusion matrix утга тайлбарлах
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix тайлбар:")
print(f"  True Negative  (TN={tn}): Үлдэнэ гэж зөв таамаглав")
print(f"  False Positive (FP={fp}): Гарна гэж буруу таамаглав")
print(f"  False Negative (FN={fn}): Гарна байхад Үлдэнэ гэв (хамгийн аюултай!)")
print(f"  True Positive  (TP={tp}): Гарна гэж зөв таамаглав")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, label=f'Decision Tree (AUC={auc:.4f})',
             color='steelblue', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'r--', label='Санамсаргүй таамаглал')
axes[1].set_title('ROC Curve', fontsize=12)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate (Recall)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Загварын үнэлгээний графикууд', fontsize=13)
plt.tight_layout()
plt.savefig('evaluation.png', dpi=150)
plt.show()
print("evaluation.png хадгалагдлаа")


# ---------------------------------------------------------------
# АЛХАМ 8: Шинэ харилцагч дээр таамаглах
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("АЛХАМ 8: Шинэ харилцагч дээр таамаглах")
print("=" * 55)

new_customers = pd.DataFrame([
    {
        'CreditScore': 620, 'Gender': 1, 'Age': 45, 'Tenure': 3,
        'Balance': 120000, 'NumOfProducts': 1, 'HasCrCard': 1,
        'IsActiveMember': 0, 'EstimatedSalary': 80000,
        'Geography_Germany': 1, 'Geography_Spain': 0
    },
    {
        'CreditScore': 750, 'Gender': 0, 'Age': 32, 'Tenure': 8,
        'Balance': 0, 'NumOfProducts': 2, 'HasCrCard': 1,
        'IsActiveMember': 1, 'EstimatedSalary': 120000,
        'Geography_Germany': 0, 'Geography_Spain': 0
    }
])

for i, row in new_customers.iterrows():
    pred = dt.predict(new_customers.iloc[[i]])[0]
    prob = dt.predict_proba(new_customers.iloc[[i]])[0][1]
    status = '⚠️  Гарна' if pred == 1 else '✅  Үлдэнэ'
    print(f"Харилцагч {i+1}: {status}  (Гарах магадлал: {prob*100:.1f}%)")


# ---------------------------------------------------------------
# НЭМЭЛТ: Оновчтой max_depth олох (CV)
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("НЭМЭЛТ: Оновчтой max_depth олох")
print("=" * 55)

from sklearn.model_selection import cross_val_score

depths = range(2, 12)
cv_scores = []
for d in depths:
    clf = DecisionTreeClassifier(
        max_depth=d, class_weight='balanced', random_state=42
    )
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean()
    cv_scores.append(score)

best_depth = depths[np.argmax(cv_scores)]
print(f"Оновчтой max_depth : {best_depth}")
print(f"Хамгийн өндөр AUC  : {max(cv_scores):.4f}")

plt.figure(figsize=(8, 4))
plt.plot(list(depths), cv_scores, marker='o', color='steelblue')
plt.axvline(best_depth, color='red', linestyle='--', label=f'Оновчтой: {best_depth}')
plt.title('Cross-Validation AUC vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('CV ROC-AUC')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('depth_tuning.png', dpi=150)
plt.show()
print("depth_tuning.png хадгалагдлаа")

print("\n" + "=" * 55)
print("Бүх гаралтын файлууд:")
print("  - eda_plots.png")
print("  - correlation_heatmap.png")
print("  - tree_plot_tree.png")
print("  - tree_rules.txt")
print("  - tree_graphviz.dot")
print("  - feature_importance.png")
print("  - evaluation.png")
print("  - depth_tuning.png")
print("=" * 55)
