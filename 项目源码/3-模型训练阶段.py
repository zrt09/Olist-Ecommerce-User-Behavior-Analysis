#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================
# 1. 导入机器学习库
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

print("库导入完成")


# In[3]:


# ============================================
# 2. 读取数据
# ============================================

df = pd.read_csv('final_model_dataset.csv')

print(df.shape)

df.head()


# In[5]:


# ============================================
# 3. 类别变量编码
# ============================================

le_payment = LabelEncoder()
le_category = LabelEncoder()

df['favorite_payment'] = le_payment.fit_transform(
    df['favorite_payment'].astype(str)
)

df['favorite_category'] = le_category.fit_transform(
    df['favorite_category'].astype(str)
)

print("类别变量编码完成")


# In[7]:


# ============================================
# 4. 特征与标签
# ============================================

X = df.drop(
    columns=[
        'customer_unique_id',
        'repurchase_label'
    ]
)

y = df['repurchase_label']

print(X.shape)
print(y.shape)


# In[9]:


# ============================================
# 5. 数据集划分
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("训练集:", X_train.shape)
print("测试集:", X_test.shape)

print("\n训练集标签分布:")
print(y_train.value_counts(normalize=True))

print("\n测试集标签分布:")
print(y_test.value_counts(normalize=True))


# In[11]:


# ============================================
# 类别分布可视化
# ============================================

plt.figure(figsize=(6,4))

sns.countplot(
    x=y,
    palette='Set2'
)

plt.title('Repurchase Label Distribution')
plt.xlabel('Repurchase Label')
plt.ylabel('Count')

for p in plt.gca().patches:
    plt.text(
        p.get_x()+0.3,
        p.get_height()+50,
        str(int(p.get_height()))
    )

plt.show()


# In[13]:


# ============================================
# 训练集测试集比例
# ============================================

sizes = [
    len(X_train),
    len(X_test)
]

labels = [
    'Train',
    'Test'
]

plt.figure(figsize=(5,5))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title('Train/Test Split')

plt.show()


# In[15]:


# ============================================
# 6. 特征标准化
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("标准化完成")


# In[17]:


# ============================================
# 计算 scale_pos_weight
# ============================================

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print(
    "scale_pos_weight:",
    scale_pos_weight
)


# In[19]:


# ============================================
# 7. 逻辑回归（Logistic Regression）
# ============================================

lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear']
}

grid_lr = GridSearchCV(
    lr,
    param_grid_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_lr.fit(X_train_scaled, y_train)

best_lr = grid_lr.best_estimator_

print("LR最佳参数:", grid_lr.best_params_)


# In[21]:


# ============================================
# 8. 随机森林（Random Forest）
# ============================================

rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(
    rf,
    param_grid_rf,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("RF最佳参数:", grid_rf.best_params_)

# ============================================
# RF参数搜索热力图
# ============================================

cv_results_rf = pd.DataFrame(
    grid_rf.cv_results_
)

pivot_table = cv_results_rf.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators'
)

plt.figure(figsize=(8,6))

sns.heatmap(
    pivot_table,
    annot=True,
    fmt='.4f',
    cmap='YlGnBu'
)

plt.title(
    'RF Hyperparameter Search Heatmap'
)

plt.xlabel(
    'n_estimators'
)

plt.ylabel(
    'max_depth'
)

plt.show()



best_rf = grid_rf.best_estimator_


# In[23]:


# ============================================
# 9. XGBoost
# ============================================

xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

param_grid_xgb = {
    'n_estimators': [100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

grid_xgb = GridSearchCV(
    xgb,
    param_grid_xgb,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)

print("XGB最佳参数:", grid_xgb.best_params_)

# XGBoost参数搜索热力图
cv_results_xgb = pd.DataFrame(
    grid_xgb.cv_results_
)

pivot_table = cv_results_xgb.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_learning_rate'
)

plt.figure(figsize=(8,6))

sns.heatmap(
    pivot_table,
    annot=True,
    fmt='.4f',
    cmap='YlGnBu'
)

plt.title(
    'XGBoost Hyperparameter Search'
)

plt.xlabel(
    'learning_rate'
)

plt.ylabel(
    'max_depth'
)

plt.show()

best_xgb = grid_xgb.best_estimator_


# In[25]:


# ============================================
# 10. 模型评估函数
# ============================================

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n========== {model_name} ==========")

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("AUC      :", round(auc, 4))

    return y_pred, y_prob


# In[27]:


# ============================================
# 11. 模型评估
# ============================================

lr_pred, lr_prob = evaluate_model(
    best_lr,
    X_test_scaled,
    y_test,
    'Logistic Regression'
)

rf_pred, rf_prob = evaluate_model(
    best_rf,
    X_test,
    y_test,
    'Random Forest'
)

xgb_pred, xgb_prob = evaluate_model(
    best_xgb,
    X_test,
    y_test,
    'XGBoost'
)


# In[29]:


# ============================================
# 模型性能对比
# ============================================

result_df = pd.DataFrame({
    'Model': [
        'LR',
        'RF',
        'XGB'
    ],
    'Accuracy': [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred)
    ],
    'F1': [
        f1_score(y_test, lr_pred),
        f1_score(y_test, rf_pred),
        f1_score(y_test, xgb_pred)
    ],
    'AUC': [
        roc_auc_score(y_test, lr_prob),
        roc_auc_score(y_test, rf_prob),
        roc_auc_score(y_test, xgb_prob)
    ]
})

print(result_df)

plt.figure(figsize=(8,5))

result_df.set_index('Model')[
    ['Accuracy','F1','AUC']
].plot(
    kind='bar'
)

plt.title('Model Performance Comparison')

plt.ylabel('Score')

plt.xticks(rotation=0)

plt.show()


# In[31]:


# ============================================
# 12. ROC曲线
# ============================================

plt.figure(figsize=(8,6))

for prob, name in [
    (lr_prob, 'Logistic Regression'),
    (rf_prob, 'Random Forest'),
    (xgb_prob, 'XGBoost')
]:

    fpr, tpr, _ = roc_curve(y_test, prob)

    auc_score = roc_auc_score(
        y_test,
        prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f'{name} (AUC={auc_score:.3f})'
    )

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()

plt.show()


# In[33]:


# ============================================
# 13. Precision-Recall Curve
# ============================================

from sklearn.metrics import precision_recall_curve

plt.figure(figsize=(8,6))

for prob,name in [
    (lr_prob,'LR'),
    (rf_prob,'RF'),
    (xgb_prob,'XGB')
]:

    precision, recall, _ = precision_recall_curve(
        y_test,
        prob
    )

    plt.plot(
        recall,
        precision,
        label=name
    )

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.title('Precision-Recall Curve')

plt.legend()

plt.show()


# In[35]:


# ============================================
# 14. 混淆矩阵
# ============================================

models = [
    ('Logistic Regression', lr_pred),
    ('Random Forest', rf_pred),
    ('XGBoost', xgb_pred)
]

for name, pred in models:

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(name)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show()


# In[37]:


# ============================================
# 15. RF特征重要性
# ============================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)

plt.figure(figsize=(10,6))

plt.barh(
    importance['Feature'],
    importance['Importance']
)

plt.gca().invert_yaxis()

plt.title('RF Feature Importance')

plt.xlabel('Importance')

plt.show()


# In[39]:


# ============================================
# 16. XGBoost特征重要性
# ============================================

importance_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_xgb.feature_importances_
})

importance_xgb = importance_xgb.sort_values(
    by='Importance',
    ascending=False
)

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_xgb,
    x='Importance',
    y='Feature',
    color='seagreen'     
)

plt.title(
    'XGBoost Feature Importance'
)

plt.tight_layout()

plt.show()


# In[ ]:




