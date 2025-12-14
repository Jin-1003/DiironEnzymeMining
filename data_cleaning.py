import pandas as pd

df = pd.read_csv('uniprotkb_diiron_2025_12_13.tsv', sep='\t')
df_reviewed = df[df['Reviewed'] == 'reviewed'].copy()
print(f"已审阅数据量: {len(df_reviewed)}")

condition_name = df_reviewed['Protein names'].str.contains('oxygenase|oxidase', case=False, na=False)
condition_cofactor = df_reviewed['Cofactor'].str.contains('diiron', case=False, na=False)
df_filtered = df_reviewed[condition_name | condition_cofactor].copy()

print(f"筛选后数据量: {len(df_filtered)}")
print("\n预览前几行:")
print(df_filtered[['Entry', 'Protein names']].head(10))

df_filtered.to_csv('cleaned_HDOs_dataset.csv', index=False)
print(f"\n数据已保存为: 'cleaned_HDOs_dataset.csv'")