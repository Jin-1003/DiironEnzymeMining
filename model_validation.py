import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("🔍 开始模型验证分析...")

# 1. 重新加载数据
df = pd.read_csv('HDOs_with_features.csv')

# 创建目标变量
def is_oxidoreductase(ec_str):
    if pd.isna(ec_str):
        return 0
    ec_str = str(ec_str)
    return 1 if ec_str.startswith('1.') else 0

df['target'] = df['EC number'].apply(is_oxidoreductase)

# 2. 准备特征（排除可能泄漏的特征）
exclude_cols = ['Entry', 'Entry Name', 'Protein names', 'Gene Names', 
                'Organism', 'Sequence', 'EC number', 'Function [CC]', 
                'Cofactor', 'Keywords', 'Reviewed', 'target']

# 特别检查是否有特征直接与EC编号相关
leakage_suspects = []
for col in df.columns:
    if 'ec' in col.lower() or 'enzyme' in col.lower():
        leakage_suspects.append(col)
        exclude_cols.append(col)

if leakage_suspects:
    print(f"⚠️  检测到可能泄漏的特征: {leakage_suspects}")

feature_columns = [col for col in df.columns if col not in exclude_cols]
X = df[feature_columns].fillna(df[feature_columns].median())
y = df['target']

print(f"使用特征数量: {X.shape[1]}")
print(f"类别分布: 氧化还原酶 {y.sum()}/{len(y)} ({y.mean():.1%})")

# 3. 十折交叉验证（更严格的评估）
print("\n🔬 十折交叉验证...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用分层K折交叉验证
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')

print(f"交叉验证准确率: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
print("各折准确率:", [f"{s:.3f}" for s in cv_scores])

# 4. 特征重要性再分析
print("\n📊 训练最终模型分析特征...")
rf.fit(X_scaled, y)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 特征重要性排名:")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:25s}: {row['importance']:.4f}")

# 5. 分析分子量特征
print("\n📈 分子量特征分析:")
if 'molecular_weight' in feature_columns:
    mw_ox = df[df['target'] == 1]['molecular_weight']
    mw_non = df[df['target'] == 0]['molecular_weight']
    
    print(f"氧化还原酶平均分子量: {mw_ox.mean():.1f} ± {mw_ox.std():.1f}")
    print(f"非氧化还原酶平均分子量: {mw_non.mean():.1f} ± {mw_non.std():.1f}")
    print(f"差异显著性: {abs(mw_ox.mean() - mw_non.mean()):.1f}")

# 6. 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 交叉验证结果
axes[0].plot(range(1, 11), cv_scores, 'o-', linewidth=2)
axes[0].axhline(y=cv_scores.mean(), color='r', linestyle='--', alpha=0.7)
axes[0].fill_between(range(1, 11), 
                     cv_scores.mean() - cv_scores.std(),
                     cv_scores.mean() + cv_scores.std(),
                     alpha=0.2)
axes[0].set_xlabel('交叉验证折数')
axes[0].set_ylabel('准确率')
axes[0].set_title('十折交叉验证结果')
axes[0].set_ylim(0.5, 1.05)
axes[0].grid(True, alpha=0.3)

# 特征重要性
top10 = feature_importance.head(10)
axes[1].barh(range(10), top10['importance'][::-1])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(top10['feature'][::-1])
axes[1].set_xlabel('特征重要性')
axes[1].set_title('Top 10 特征重要性')

plt.tight_layout()
plt.savefig('model_validation_results.png', dpi=300, bbox_inches='tight')
print("\n📊 验证图表已保存: model_validation_results.png")

# 7. 结论分析
print("\n✅ 验证完成!")
print(f"平均交叉验证准确率: {cv_scores.mean():.3f}")
if cv_scores.mean() > 0.95:
    print("⚠️  注意: 准确率非常高，可能表明:")
    print("    1. 双铁金属酶中氧化还原酶有独特的序列特征")
    print("    2. 分子量和特定氨基酸频率是关键区分因素")
    print("    3. 数据集本身具有很好的可分性")