import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("🧠 开始二分类机器学习建模：预测是否为氧化还原酶")

# 1. 加载数据
df = pd.read_csv('HDOs_with_features.csv')
print(f"数据集形状: {df.shape}")

# 2. 创建二分类目标变量：是否为氧化还原酶（EC 1类）
def is_oxidoreductase(ec_str):
    """判断是否为氧化还原酶（EC 1类）"""
    if pd.isna(ec_str):
        return 0  # 没有EC编号的视为非氧化还原酶
    ec_str = str(ec_str)
    return 1 if ec_str.startswith('1.') else 0

df['is_oxidoreductase'] = df['EC number'].apply(is_oxidoreductase)
print(f"\n📊 类别分布:")
print(f"  氧化还原酶 (EC 1类): {df['is_oxidoreductase'].sum()} 条")
print(f"  非氧化还原酶: {len(df) - df['is_oxidoreductase'].sum()} 条")
print(f"  比例: {df['is_oxidoreductase'].mean():.2%}")

# 3. 准备特征
exclude_cols = ['Entry', 'Entry Name', 'Protein names', 'Gene Names', 
                'Organism', 'Sequence', 'EC number', 'Function [CC]', 
                'Cofactor', 'Keywords', 'Reviewed', 'is_oxidoreductase']

feature_columns = [col for col in df.columns if col not in exclude_cols]
X = df[feature_columns].fillna(df[feature_columns].median())
y = df['is_oxidoreductase']

print(f"\n🔧 特征工程:")
print(f"  特征数量: {X.shape[1]}")
print(f"  样本数量: {X.shape[0]}")

# 4. 处理高度相关的特征（移除冗余特征）
# 移除 'Length'（保留 'length' 和 'molecular_weight' 中的一个）
if 'Length' in feature_columns:
    feature_columns.remove('Length')
if 'length' in feature_columns:
    feature_columns.remove('length')  # 保留 'molecular_weight'
X = df[feature_columns].fillna(df[feature_columns].median())

# 5. 数据标准化和划分
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 训练随机森林模型
print("\n🌲 训练随机森林模型...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. 模型评估
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

accuracy = (y_pred == y_test).mean()
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n🎯 模型性能:")
print(f"  准确率: {accuracy:.3f}")
print(f"  ROC AUC: {roc_auc:.3f}")
print("\n📋 详细分类报告:")
print(classification_report(y_test, y_pred, target_names=['非氧化还原酶', '氧化还原酶']))

# 8. 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 最重要的10个特征:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 9. 保存结果
joblib.dump(rf_model, 'rf_binary_classifier.pkl')
joblib.dump(scaler, 'binary_scaler.pkl')
feature_importance.to_csv('binary_feature_importance.csv', index=False)

print("\n💾 已保存:")
print("  rf_binary_classifier.pkl - 随机森林模型")
print("  binary_scaler.pkl - 标准化器")
print("  binary_feature_importance.csv - 特征重要性")

# 10. 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 特征重要性
top10 = feature_importance.head(10)
axes[0].barh(range(10), top10['importance'][::-1])
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(top10['feature'][::-1])
axes[0].set_xlabel('特征重要性')
axes[0].set_title('预测氧化还原酶的关键特征')

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_xlabel('预测类别')
axes[1].set_ylabel('真实类别')
axes[1].set_xticklabels(['非氧化还原酶', '氧化还原酶'])
axes[1].set_yticklabels(['非氧化还原酶', '氧化还原酶'])
axes[1].set_title('混淆矩阵')

plt.tight_layout()
plt.savefig('binary_classification_results.png', dpi=300, bbox_inches='tight')
print("\n📊 可视化图表已保存: binary_classification_results.png")

print("\n✅ 二分类建模完成！")
print("\n📝 大作业报告要点:")
print("1. 研究问题: 从蛋白质序列特征预测其是否为氧化还原酶")
print("2. 数据来源: 368条双铁金属酶，73.6%为氧化还原酶")
print(f"3. 模型性能: 随机森林准确率{accuracy:.3f}, ROC AUC{roc_auc:.3f}")
print("4. 关键发现: 揭示了序列特征与氧化还原酶功能的关系")
print("5. 应用价值: 可用于快速注释未知蛋白的功能类别")