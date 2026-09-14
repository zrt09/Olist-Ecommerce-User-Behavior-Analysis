#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ============================================
# 1. 导入库与全局风格配置
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 统一图表视觉风格配置
sns.set_theme(style="whitegrid") 

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['figure.dpi'] = 120 
plt.rcParams['figure.figsize'] = (10, 6)  # 默认图表大小

# 设置 Seaborn 调色板
sns.set_palette("Blues_d")

print("库导入与全局风格配置完成。")


# In[5]:


# ============================================
# 2. 数据读取与基础概览
# ============================================

print("========== 开始读取数据仓库文件 ==========")

# 读取CSV文件
final_model_dataset = pd.read_csv('final_model_dataset.csv')
fact_orders = pd.read_csv('fact_orders.csv')
dim_customer = pd.read_csv('dim_customer.csv')
dim_product = pd.read_csv('dim_product.csv')
dim_payment = pd.read_csv('dim_payment.csv')
dim_time = pd.read_csv('dim_time.csv')

# 定义数据集字典以便循环输出概览
datasets = {
    'final_model_dataset': final_model_dataset,
    'fact_orders': fact_orders,
    'dim_customer': dim_customer,
    'dim_product': dim_product,
    'dim_payment': dim_payment,
    'dim_time': dim_time
}

# 输出基础信息
for name, df in datasets.items():
    print(f"\n数据集: {name}")
    print(f"-> 数据规模 (行数, 列数): {df.shape}")
    print(f"-> 字段列表: {list(df.columns)}")
    print("-> 前2行样例:")
    display(df.head(2))

print("\n========== 数据仓库文件读取与概览完成 ==========")


# In[11]:


# ============================================
# 3. 星型模型OLAP多维分析
# 3.1 月度订单量与销售额趋势分析
# ============================================

# 关联事实表与时间维度表
olap_monthly = fact_orders.copy()
olap_monthly = olap_monthly.merge(
    dim_time, 
    left_on='order_purchase_timestamp', 
    right_on='purchase_time', 
    how='left'
)

# 按年、月聚合订单量与销售额
monthly_trend = olap_monthly.groupby(['year', 'month']).agg(
    order_count=('order_id', 'nunique'),
    total_sales=('payment_value', 'sum')
).reset_index()

# 过滤或排序，创建平滑的时间轴标签
monthly_trend['year_month'] = monthly_trend['year'].astype(str) + '-' + monthly_trend['month'].astype(str).str.zfill(2)
monthly_trend = monthly_trend.sort_values('year_month').reset_index(drop=True)

# 剔除数据不全的边缘月份（根据实际Olist数据集生命周期，2016和2018年部分月份数据极为稀疏，此处保证连续性）
monthly_trend = monthly_trend[(monthly_trend['year_month'] >= '2017-01') & (monthly_trend['year_month'] <= '2018-08')]

# 绘制双轴图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 右轴：销售额柱状图
ax2 = ax1.twinx()
sns.barplot(x='year_month', y='total_sales', data=monthly_trend, alpha=0.6, color='steelblue', ax=ax2)
ax2.set_ylabel('月总销售额 (BRL)', color='steelblue', fontsize=12)
ax2.tick_params(axis='y', labelcolor='steelblue')
ax2.grid(False) # 隐藏右轴网格线避免冲突

# 左轴：订单量折线图
sns.lineplot(x='year_month', y='order_count', data=monthly_trend, marker='o', color='darkorange', linewidth=2.5, ax=ax1, label='订单量')
ax1.set_ylabel('月总订单量 (单)', color='darkorange', fontsize=12)
ax1.tick_params(axis='y', labelcolor='darkorange')
ax1.set_xlabel('年份-月份', fontsize=12)
ax1.set_xticklabels(monthly_trend['year_month'], rotation=45)

plt.title('Olist 电商平台月度订单量与销售额演变趋势 (OLAP多维分析)', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
ax1.legend(loc='upper left')
plt.annotate('数据来源: Olist 巴西电商数据集', xy=(0.02, 0.02), xycoords='figure fraction', fontsize=9, color='gray')
plt.show()


# In[15]:


# ============================================
# 3.2 商品类别销售额Top10
# ============================================

# 关联事实表与商品维度表
if 'payment_value' not in fact_orders.columns:
    olap_product = fact_orders.merge(dim_payment[['order_id', 'payment_value']], on='order_id', how='left')
else:
    olap_product = fact_orders.copy()

olap_product = olap_product.merge(dim_product, on='product_id', how='left')

# 按英文类目聚合销售额
product_sales = olap_product.groupby('product_category_name_english')['payment_value'].sum().reset_index()
product_top10 = product_sales.sort_values(by='payment_value', ascending=False).head(10)

# 绘制水平条形图
plt.figure(figsize=(11, 6))
sns.barplot(x='payment_value', y='product_category_name_english', data=product_top10, palette='Blues_r')

plt.title('Olist 电商平台商品类别销售额 Top 10', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('总销售额', fontsize=12)
plt.ylabel('商品类目', fontsize=12)
plt.tight_layout()
plt.show()


# In[17]:


# ============================================
# 3.3 客户州订单量Top10
# ============================================

# 关联事实表与客户维度表
olap_customer = fact_orders.merge(dim_customer, on='customer_id', how='left')

# 按客户所在州统计订单数
state_orders = olap_customer.groupby('customer_state')['order_id'].nunique().reset_index()
state_top10 = state_orders.sort_values(by='order_id', ascending=False).head(10)

# 绘制条形图
plt.figure(figsize=(11, 5))
sns.barplot(x='customer_state', y='order_id', data=state_top10, palette='GnBu_r')

plt.title('Olist 电商平台客户所在州订单量 Top 10', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('巴西省份/州简称', fontsize=12)
plt.ylabel('总订单量 (单)', fontsize=12)
plt.tight_layout()
plt.show()


# In[21]:


# ============================================
# 3.4 支付方式与月份趋势分析（平均订单金额）
# ============================================

# 关联事实表、支付维度表与时间维度表
olap_payment = fact_orders.copy()
olap_payment = olap_payment.merge(dim_time, left_on='order_purchase_timestamp', right_on='purchase_time', how='left')

# 创建时间轴标签
olap_payment['year_month'] = olap_payment['year'].astype(str) + '-' + olap_payment['month'].astype(str).str.zfill(2)
olap_payment = olap_payment[(olap_payment['year_month'] >= '2017-01') & (olap_payment['year_month'] <= '2018-08')]

# 按支付方式和月份统计平均订单金额
payment_monthly = olap_payment.groupby(['payment_type', 'year_month'])['payment_value'].mean().reset_index()

# 过滤掉样本极少的支付方式(如not_defined)
payment_monthly = payment_monthly[payment_monthly['payment_type'].isin(['credit_card', 'boleto', 'voucher', 'debit_card'])]

# 绘制多折线图
plt.figure(figsize=(12, 6))
sns.lineplot(x='year_month', y='payment_value', hue='payment_type', data=payment_monthly, marker='o', linewidth=2)

plt.title('不同支付方式的月度平均订单金额趋势变化', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('年份-月份', fontsize=12)
plt.ylabel('平均订单金额', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='支付方式')
plt.tight_layout()
plt.show()


# In[25]:


# ============================================
# 4. RFM 用户价值分析
# 4.1 构建 RFM 指标
# ============================================

print("========== 开始构建 RFM 模型 ==========")

# 1. 从预处理模型数据集中提取唯一用户标识以及 F 与 M
rfm_base = final_model_dataset[['customer_unique_id', 'purchase_frequency', 'total_spending']].copy()
rfm_base.columns = ['customer_unique_id', 'Frequency', 'Monetary']

# 2. 关联事实表与客户维度表
customer_orders = fact_orders.merge(dim_customer, on='customer_id', how='left')

# 3. 关联时间维度表：
customer_orders = customer_orders.merge(
    dim_time, 
    left_on='order_purchase_timestamp', 
    right_on='purchase_time', 
    how='left'
)

# 4. 将时间维度表中的 year, month, day 拼接并转换为 datetime 格式
customer_orders['order_date'] = pd.to_datetime(
    customer_orders['year'].astype(str) + '-' + 
    customer_orders['month'].astype(str) + '-' + 
    customer_orders['day'].astype(str)
)

# 5. 获取每个唯一用户（customer_unique_id）的最后一次购买日期
last_purchase = customer_orders.groupby('customer_unique_id')['order_date'].max().reset_index()
last_purchase.columns = ['customer_unique_id', 'last_purchase_date']

# 6. 将最后一次购买日期合并到 RFM 基础表
rfm_df = rfm_base.merge(last_purchase, on='customer_unique_id', how='left')

# 7. 设定观察截止日期
cutoff_date = pd.to_datetime('2018-03-01')

# 8. 计算 Recency (当前截止日期距离最后一次购买的天数)
rfm_df['Recency'] = (cutoff_date - rfm_df['last_purchase_date']).dt.days

# 9. 处理未来数据：若在截止日期后仍有购买，Recency会为负数，此类用户定义为极度活跃(赋值为0)
rfm_df['Recency'] = rfm_df['Recency'].apply(lambda x: 0 if x < 0 else x)

# 10. 筛选最终需要的 RFM 核心列
rfm_final = rfm_df[['customer_unique_id', 'Recency', 'Frequency', 'Monetary']]

print("\n-> RFM 表前5行样例:")
display(rfm_final.head())

print("\n-> RFM 表描述性统计信息:")
display(rfm_final.describe())

print("\n========== RFM 模型指标构建完成 ==========")


# In[27]:


# ============================================
# 4.2 RFM 分布可视化分析
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recency 分布
sns.histplot(rfm_final['Recency'], bins=30, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Recency (最近一次购买间隔) 分布', fontsize=12, fontweight='bold')
axes[0].set_xlabel('天数')

# Frequency 分布（电商常态：绝大多数为1，呈长尾偏态，应用对数刻度）
sns.histplot(rfm_final['Frequency'], bins=15, kde=False, ax=axes[1], color='coral')
axes[1].set_title('Frequency (购买频次) 分布 [对数轴]', fontsize=12, fontweight='bold')
axes[1].set_xlabel('次')
axes[1].set_yscale('log')

# Monetary 分布（长尾偏态，应用对数刻度）
sns.histplot(rfm_final['Monetary'], bins=40, kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title('总消费金额)分布', fontsize=12, fontweight='bold')
axes[2].set_xlabel('金额 ')
axes[2].set_xscale('log')

plt.suptitle('RFM 指标原始分布图谱', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[29]:


# ============================================
# 5. 用户聚类分析
# 5.1 肘部法确定最佳聚类数 K
# ============================================

# 提取特征进行标准化
X_rfm = rfm_final[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rfm)

# 循环计算 K=2 到 8 的 SSE
sse = []
k_range = range(2, 9)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# 绘制肘部曲线
plt.figure(figsize=(9, 5))
plt.plot(k_range, sse, marker='o', linestyle='--', color='darkblue', linewidth=2)
plt.title('通过肘部法确定 KMeans 最佳聚类数', fontsize=14, fontweight='bold')
plt.xlabel('聚类簇数 (K)', fontsize=12)
plt.ylabel('簇内误方差 (SSE / Inertia)', fontsize=12)
plt.axvline(x=4, color='red', linestyle=':', label='选定最佳 K = 4')
plt.legend()
plt.tight_layout()
plt.show()


# In[43]:


# ============================================
# 5.2 KMeans聚类
# 5.3 PCA 降维与聚类空间可视化
# ============================================

# 选定 K=4 执行聚类
best_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_final['Cluster'] = best_kmeans.fit_transform(X_scaled).argmin(axis=1) # 获取对应类别标签
rfm_final['Cluster'] = best_kmeans.labels_

# PCA降维到二维空间
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 将降维结果合并至 DataFrame
rfm_pca = pd.DataFrame(data=X_pca, columns=['Component_1', 'Component_2'])
rfm_pca['Cluster'] = rfm_final['Cluster']

# 获取 PCA 空间下的聚类中心
centers_scaled = best_kmeans.cluster_centers_
centers_pca = pca.transform(centers_scaled)

# 绘制二维降维散射图
plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(
    x='Component_1', y='Component_2', 
    hue='Cluster', palette='Set1', 
    data=rfm_pca, alpha=0.5, edgecolor=None
)

# 标出聚类中心
plt.scatter(
    centers_pca[:, 0], centers_pca[:, 1], 
    s=250, marker='X', color='black', 
    edgecolor='white', linewidth=2, label='聚类中心'
)

plt.title('基于 RFM 标准化数据的 KMeans 聚类空间图谱 (PCA降维可视化)', fontsize=14, fontweight='bold')
plt.xlabel('主成分 1', fontsize=12)
plt.ylabel('主成分 2', fontsize=12)
plt.legend(title='用户群体/簇')
plt.tight_layout()
plt.show()


# In[53]:


# ============================================
# 6. 用户分层分析与画像构建
# ============================================

# 按照聚类统计各大指标均值和占比
cluster_summary = rfm_final.groupby('Cluster').agg(
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean'),
    User_Count=('customer_unique_id', 'count')
).reset_index()

# 计算用户占比
total_users = cluster_summary['User_Count'].sum()
cluster_summary['User_Ratio (%)'] = (cluster_summary['User_Count'] / total_users * 100).round(2)

# 高Monetary为高价值；小Recency为活跃；大Recency为沉睡
def label_clusters(row):
    if row['Avg_Frequency'] > 1.5 and row['Avg_Monetary'] > 300:
        return '高频次用户'
    elif row['Avg_Frequency'] <= 1.5 and row['Avg_Monetary'] > 300:
        return '大额客单用户'
    elif row['Avg_Recency'] < 100:
        return '新客与潜力活跃用户'          
    elif row['Avg_Recency'] >= 250:
        return '沉睡用户'          
    else:
        return '常规维持期大众用户'

cluster_summary['Segment_Name'] = cluster_summary.apply(label_clusters, axis=1)

print("========== RFM 用户分层多维全景汇总表 ==========")
display(cluster_summary.sort_values(by='Avg_Monetary', ascending=False))


# In[ ]:




