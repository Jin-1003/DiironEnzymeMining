import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re

# 1. 读取清洗后的数据
df = pd.read_csv('cleaned_HDOs_dataset.csv')
print(f"开始处理 {len(df)} 条蛋白质序列...")

# 2. 基础序列特征提取函数
def extract_sequence_features(seq):
    """提取氨基酸频率和简单序列特征"""
    if pd.isna(seq):
        return {}
    
    seq = str(seq).upper()
    
    # 基础特征
    features = {
        'length': len(seq),
    }
    
    # 20种标准氨基酸的频率
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for aa in amino_acids:
        features[f'freq_{aa}'] = seq.count(aa) / len(seq) if len(seq) > 0 else 0
    
    # 氨基酸类别组成
    hydrophobic = 'AVILMFYW'  # 疏水性
    hydrophilic = 'DEKR'      # 亲水性
    charged = 'DEKRH'         # 带电
    polar = 'STYCNQ'          # 极性
    special = 'CGPH'          # 特殊
    
    categories = {
        'hydrophobic': hydrophobic,
        'hydrophilic': hydrophilic,
        'charged': charged,
        'polar': polar,
        'special': special
    }
    
    for cat_name, cat_aas in categories.items():
        count = sum(seq.count(aa) for aa in cat_aas)
        features[f'frac_{cat_name}'] = count / len(seq) if len(seq) > 0 else 0
    
    return features

# 3. 理化性质特征提取（使用BioPython）
def extract_physicochemical_features(seq):
    """提取理化性质特征"""
    if pd.isna(seq) or len(str(seq)) < 10:
        return {}
    
    try:
        seq_str = str(seq).upper()
        # 移除非标准氨基酸字符
        seq_str = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq_str)
        
        if len(seq_str) < 10:
            return {}
            
        protein = ProteinAnalysis(seq_str)
        
        features = {
            'molecular_weight': protein.molecular_weight(),
            'aromaticity': protein.aromaticity(),
            'instability_index': protein.instability_index(),
            'isoelectric_point': protein.isoelectric_point(),
            'gravy': protein.gravy(),  # 平均疏水性
            'flexibility_mean': np.mean(protein.flexibility()) if hasattr(protein, 'flexibility') else 0,
        }
        
        # 二级结构倾向
        sec_struct = protein.secondary_structure_fraction()
        features.update({
            'helix_fraction': sec_struct[0],
            'turn_fraction': sec_struct[1],
            'sheet_fraction': sec_struct[2]
        })
        
        return features
    except Exception as e:
        print(f"处理序列时出错: {str(e)[:50]}...")
        return {}

# 4. 主处理流程
print("正在提取序列组成特征...")
seq_features_list = []
for seq in df['Sequence']:
    seq_features_list.append(extract_sequence_features(seq))

print("正在提取理化性质特征...")
physico_features_list = []
for seq in df['Sequence']:
    physico_features_list.append(extract_physicochemical_features(seq))

# 5. 合并所有特征
df_seq_features = pd.DataFrame(seq_features_list)
df_physico_features = pd.DataFrame(physico_features_list)

# 合并所有特征
df_features = pd.concat([df, df_seq_features, df_physico_features], axis=1)

# 6. 保存特征数据集
output_file = 'HDOs_with_features.csv'
df_features.to_csv(output_file, index=False)

print(f"\n✅ 特征提取完成！")
print(f"原始特征数量: {df.shape[1]} 列")
print(f"新特征数量: {df_features.shape[1]} 列")
print(f"新增特征: {df_features.shape[1] - df.shape[1]} 个")
print(f"总数据条数: {len(df_features)}")
print(f"已保存到: {output_file}")

# 7. 显示特征摘要
print("\n📊 特征摘要:")
print("-" * 40)
print("1. 基础序列特征:")
print(f"   • 序列长度 (length)")
print(f"   • 20种氨基酸频率 (freq_A 到 freq_Y)")
print(f"   • 5类氨基酸比例 (frac_hydrophobic 等)")

print("\n2. 理化性质特征:")
print(f"   • 分子量 (molecular_weight)")
print(f"   • 等电点 (isoelectric_point)")
print(f"   • 芳香性 (aromaticity)")
print(f"   • 不稳定指数 (instability_index)")
print(f"   • 平均疏水性 (gravy)")
print(f"   • 二级结构比例 (helix_fraction 等)")