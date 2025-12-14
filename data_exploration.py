import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 加载特征数据
print("📂 加载特征数据...")
df = pd.read_csv('HDOs_with_features.csv')
print(f"数据集形状: {df.shape}")
print(f"特征数量: {df.shape[1] - 12} 个数值特征")  # 减去原始12列

# 2. 检查缺失值
print("\n🔍 检查缺失值情况:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_data,
    '缺失比例%': missing_percent
})
# 只显示有缺失值的列
missing_with_values = missing_df[missing_df['缺失数量'] > 0]
if len(missing_with_values) > 0:
    print(missing_with_values.sort_values('缺失数量', ascending=False).head(10))
else:
    print("✅ 没有缺失值！")

# 3. 处理缺失值（用中位数填充数值列）
print("\n🔄 处理缺失值...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print("✅ 缺失值已用中位数填充")

# 4. 特征分布可视化
print("\n📈 绘制特征分布图...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('关键特征分布', fontsize=16)

# 选择几个关键特征展示
key_features = ['length', 'molecular_weight', 'isoelectric_point', 
                'gravy', 'instability_index', 'helix_fraction']

for idx, feature in enumerate(key_features):
    if feature in df.columns:
        ax = axes[idx//3, idx%3]
        ax.hist(df[feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(feature)
        ax.set_ylabel('频数')
        ax.set_title(f'{feature} 分布')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
print("✅ 特征分布图已保存为 'feature_distributions.png'")

# 5. 特征相关性分析
print("\n📊 计算特征相关性...")
# 选择数值特征进行相关性分析（排除非数值列和ID列）
exclude_cols = ['Entry', 'Entry Name', 'Protein names', 'Gene Names', 
                'Organism', 'Sequence', 'EC number', 'Function [CC]', 
                'Cofactor', 'Keywords', 'Reviewed']
numeric_features = [col for col in df.columns if col not in exclude_cols]

if len(numeric_features) > 0:
    corr_matrix = df[numeric_features].corr()
    
    # 绘制相关性热图（只显示前20个特征的相关性，避免图像太密集）
    features_for_heatmap = numeric_features[:20]
    corr_subset = df[features_for_heatmap].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('特征相关性热图 (前20个特征)')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    print("✅ 相关性热图已保存为 'feature_correlation.png'")
    
    # 找出高度相关的特征对
    print("\n🔗 高度相关的特征对 (|相关性| > 0.8):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        for pair in high_corr_pairs[:10]:  # 只显示前10对
            print(f"  {pair[0]} 与 {pair[1]}: {pair[2]:.3f}")
    else:
        print("  没有高度相关的特征对")

# 6. 基于EC编号的功能分类探索
print("\n🔬 基于EC编号的功能分析...")
if 'EC number' in df.columns:
    # 提取EC编号的第一部分（大类）
    df['EC_class'] = df['EC number'].astype(str).str.extract(r'^(\d+)\.')
    
    ec_counts = df['EC_class'].value_counts()
    print("EC编号大类分布:")
    for ec_class, count in ec_counts.items():
        if pd.notna(ec_class):
            ec_names = {
                '1': '氧化还原酶',
                '2': '转移酶', 
                '3': '水解酶',
                '4': '裂合酶',
                '5': '异构酶',
                '6': '连接酶'
            }
            name = ec_names.get(ec_class, '未知')
            print(f"  EC {ec_class}.x.x.x ({name}): {count} 条")

# 7. 保存处理后的数据
output_file = 'HDOs_processed_ready.csv'
# 移除原始的大文本列，保留特征和关键信息
cols_to_keep = ['Entry', 'Protein names', 'EC number'] + numeric_features
df_clean = df[cols_to_keep].copy()
df_clean.to_csv(output_file, index=False)

print(f"\n🎉 数据探索完成！")
print(f"✅ 处理后的数据已保存: {output_file}")
print(f"✅ 可视化图表已保存")
print(f"\n📋 下一步建议:")
print("1. 检查 'feature_distributions.png' 了解特征分布")
print("2. 查看 'feature_correlation.png' 识别相关特征")
print("3. 使用 'HDOs_processed_ready.csv' 进行机器学习建模")

# 显示前几行处理后的数据
print(f"\n📄 处理后的数据预览:")
print(df_clean[['Entry', 'Protein names', 'length', 'molecular_weight', 'isoelectric_point']].head())