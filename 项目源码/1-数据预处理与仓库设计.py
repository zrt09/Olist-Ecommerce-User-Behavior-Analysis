#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================
# 1. 导入库
# ============================================

import pandas as pd
import numpy as np

from datetime import timedelta

# 显示所有列
pd.set_option('display.max_columns', None)

print("库导入完成")


# In[3]:


# ============================================
# 2. 读取数据
# ============================================

orders = pd.read_csv('olist_orders_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
category_translation = pd.read_csv('product_category_name_translation.csv')

print("数据读取完成")

print("Orders shape:", orders.shape)
print("Customers shape:", customers.shape)
print("Order Items shape:", order_items.shape)
print("Products shape:", products.shape)
print("Payments shape:", payments.shape)


# In[5]:


# ============================================
# 2.1 探索性数据分析（EDA）
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

sns.set_context(
    "notebook",
    font_scale=1.2
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['figure.dpi'] = 120


# In[7]:


# 图1 订单状态分布

status_counts = orders['order_status'].value_counts()

plt.figure(figsize=(10,6))

ax = sns.barplot(
    x=status_counts.values,
    y=status_counts.index,
    palette='Blues_r'
)

for i, v in enumerate(status_counts.values):
    ax.text(
        v + 500,
        i,
        f'{v:,}',
        va='center'
    )

plt.title(
    '订单状态分布',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('订单数量')
plt.ylabel('订单状态')

plt.tight_layout()
plt.show()


# In[9]:


# 图2 月度订单趋势
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp']
)

monthly_orders = (
    orders
    .groupby(
        orders['order_purchase_timestamp'].dt.to_period('M')
    )
    .size()
)

plt.figure(figsize=(12,5))

monthly_orders.plot(marker='o')

plt.title('按月订单量趋势')
plt.xlabel('月份')
plt.ylabel('订单数')

plt.grid(True)

plt.show()


# In[11]:


# 图3 支付金额分布
plt.figure(figsize=(10,6))

payment_clean = payments[
    payments['payment_value'] <=
    payments['payment_value'].quantile(0.99)
]

sns.histplot(
    payment_clean['payment_value'],
    bins=50,
    kde=True
)

plt.title(
    '订单支付金额分布（99%分位内）',
    fontsize=16
)

plt.xlabel('支付金额')
plt.ylabel('订单数')

plt.tight_layout()

plt.show()


# In[15]:


# 图4 缺失率排序条形图
missing_rate = pd.concat([
    orders.isnull().mean(),
    products.isnull().mean()
])

missing_rate = (
    missing_rate[missing_rate > 0]
    .sort_values()
)

plt.figure(figsize=(10,6))

bars = plt.barh(
    missing_rate.index,
    missing_rate.values * 100
)

for bar in bars:

    plt.text(
        bar.get_width() + 0.05,
        bar.get_y() + bar.get_height()/2,
        f'{bar.get_width():.2f}%',
        va='center'
    )

plt.xlabel('缺失率 (%)')

plt.title(
    '数据字段缺失率分布',
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()

plt.show()


# In[17]:


# ============================================
# 3.1 时间字段转换
# ============================================

time_columns = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in time_columns:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')

print("时间字段转换完成")

orders[time_columns].head()


# In[19]:


# ============================================
# 3.2 缺失值处理
# ============================================

# 查看缺失值
print("\nOrders 缺失值:")
print(orders.isnull().sum())

print("\nProducts 缺失值:")
print(products.isnull().sum())


# products表数值字段使用中位数填充
product_numeric_cols = [
    'product_name_lenght',
    'product_description_lenght',
    'product_photos_qty',
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm'
]

for col in product_numeric_cols:
    median_value = products[col].median()
    products[col].fillna(median_value, inplace=True)

# 商品类别缺失填充
products['product_category_name'] = products[
    'product_category_name'
].fillna('unknown_category')


# 删除已送达但缺失送达时间的订单
orders = orders[
    ~(
        (orders['order_status'] == 'delivered') &
        (orders['order_delivered_customer_date'].isnull())
    )
]

print("缺失值处理完成")


# In[21]:


# ============================================
# 3.3 重复值处理
# ============================================

print("Orders重复值:", orders.duplicated().sum())
print("Customers重复值:", customers.duplicated().sum())
print("Order Items重复值:", order_items.duplicated().sum())
print("Payments重复值:", payments.duplicated().sum())

# 删除重复值
orders.drop_duplicates(inplace=True)
customers.drop_duplicates(inplace=True)
order_items.drop_duplicates(inplace=True)
payments.drop_duplicates(inplace=True)

print("重复值处理完成")


# In[23]:


# ============================================
# 3.4 异常值处理
# ============================================

# 删除负价格与负运费
order_items = order_items[
    (order_items['price'] >= 0) &
    (order_items['freight_value'] >= 0)
]

# IQR方法检测价格异常值
Q1 = order_items['price'].quantile(0.25)
Q3 = order_items['price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

order_items = order_items[
    (order_items['price'] >= lower_bound) &
    (order_items['price'] <= upper_bound)
]

print("异常值处理完成")

print(order_items[['price', 'freight_value']].describe())


# In[25]:


# ============================================
# 3.5 时间逻辑检查
# ============================================

orders = orders[
    (
        orders['order_approved_at'].isnull()
    ) |
    (
        orders['order_purchase_timestamp']
        <= orders['order_approved_at']
    )
]

orders = orders[
    (
        orders['order_delivered_customer_date'].isnull()
    ) |
    (
        orders['order_delivered_carrier_date']
        <= orders['order_delivered_customer_date']
    )
]

print("订单时间逻辑检查完成")
print("当前订单数:", orders.shape[0])


# In[27]:


# ============================================
# 4.1 商品类别翻译
# ============================================

products = products.merge(
    category_translation,
    how='left',
    on='product_category_name'
)

print("商品类别翻译完成")

products.head()


# In[29]:


# ============================================
# 4.2 构建订单宽表
# ============================================

# 订单 + 客户
df = orders.merge(
    customers,
    on='customer_id',
    how='left'
)

# 订单 + 商品
df = df.merge(
    order_items,
    on='order_id',
    how='left'
)

# 商品信息
df = df.merge(
    products,
    on='product_id',
    how='left'
)

# 支付信息
df = df.merge(
    payments,
    on='order_id',
    how='left'
)

print("宽表构建完成")

print("宽表维度:", df.shape)

df.head()


# In[31]:


# ============================================
# 5. 数据仓库设计
# ============================================

# 时间维度表
dim_time = pd.DataFrame()

dim_time['purchase_time'] = df['order_purchase_timestamp']

dim_time['year'] = dim_time['purchase_time'].dt.year
dim_time['month'] = dim_time['purchase_time'].dt.month
dim_time['day'] = dim_time['purchase_time'].dt.day
dim_time['weekday'] = dim_time['purchase_time'].dt.weekday
dim_time['hour'] = dim_time['purchase_time'].dt.hour

# 去重
dim_time.drop_duplicates(inplace=True)

# 添加代理主键
dim_time = dim_time.reset_index(drop=True)

dim_time['time_id'] = (
    dim_time.index + 1
)

# 调整列顺序
dim_time = dim_time[
    [
        'time_id',
        'purchase_time',
        'year',
        'month',
        'day',
        'weekday',
        'hour'
    ]
]

# 客户维度表
dim_customer = customers[
    [
        'customer_id',
        'customer_unique_id',
        'customer_city',
        'customer_state'
    ]
].drop_duplicates()


# 产品维度表
dim_product = products[
    [
        'product_id',
        'product_category_name_english',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm'
    ]
].drop_duplicates()


# 支付维度表
dim_payment = payments[
    [
        'payment_type',
        'payment_installments'
    ]
].drop_duplicates()


# 订单事实表
fact_orders = df[
    [
        'order_id',
        'customer_id',
        'product_id',
        'order_purchase_timestamp',
        'price',
        'freight_value',
        'payment_value',
        'payment_type',
        'payment_installments'
    ]
]

print("星型模型构建完成")

print("fact_orders:", fact_orders.shape)
print("dim_customer:", dim_customer.shape)
print("dim_product:", dim_product.shape)
print("dim_payment:", dim_payment.shape)
print("dim_time:", dim_time.shape)


# In[33]:


# ============================================
# 时间切分
# ============================================

# 设置时间切分点
cutoff_date = pd.to_datetime('2018-03-01')

# 标签窗口结束时间
label_end_date = pd.to_datetime('2018-08-31')

# --------------------------------
# 历史数据（用于构造特征）
# --------------------------------

history_df = df[
    df['order_purchase_timestamp'] < cutoff_date
].copy()

# --------------------------------
# 未来窗口数据（用于构造标签）
# --------------------------------

future_df = df[
    (
        df['order_purchase_timestamp'] >= cutoff_date
    ) &
    (
        df['order_purchase_timestamp'] <= label_end_date
    )
].copy()

print("历史数据量:", history_df.shape)
print("未来窗口数据量:", future_df.shape)


# ============================================
# 6. 特征工程
# ============================================

# 用户订单时间排序
history_df = history_df.sort_values(
    by=[
        'customer_unique_id',
        'order_purchase_timestamp'
    ]
)

# 用户订单次数
purchase_frequency = history_df.groupby(
    'customer_unique_id'
)['order_id'].nunique()

# 用户平均订单金额
avg_order_value = history_df.groupby(
    'customer_unique_id'
)['payment_value'].mean()

# 用户总消费金额
total_spending = history_df.groupby(
    'customer_unique_id'
)['payment_value'].sum()

# 最近购买时间
last_purchase = history_df.groupby(
    'customer_unique_id'
)['order_purchase_timestamp'].max()

# 最早购买时间
first_purchase = history_df.groupby(
    'customer_unique_id'
)['order_purchase_timestamp'].min()

# 生命周期
customer_lifetime = (
    last_purchase - first_purchase
).dt.days

# 平均购买间隔
purchase_interval = (
    customer_lifetime /
    (purchase_frequency - 1)
)

purchase_interval = purchase_interval.replace(
    [np.inf, -np.inf],
    365
).fillna(365)

print("用户基础特征完成")


# In[35]:


# ============================================
# 6.2 支付行为特征
# ============================================

# 平均分期数
avg_installments = history_df.groupby(
    'customer_unique_id'
)['payment_installments'].mean()

# 最常用支付方式
favorite_payment = history_df.groupby(
    'customer_unique_id'
)['payment_type'].agg(
    lambda x: x.mode().iloc[0]
    if not x.mode().empty
    else 'unknown'
)

print("支付行为特征完成")


# In[37]:


# ============================================
# 6.3 商品偏好特征
# ============================================

favorite_category = history_df.groupby(
    'customer_unique_id'
)['product_category_name_english'].agg(
    lambda x: x.mode().iloc[0]
    if not x.mode().empty
    else 'unknown'
)

print("商品偏好特征完成")


# In[39]:


# ============================================
# 6.4 物流特征
# ============================================

history_df['delivery_days'] = (
    history_df['order_delivered_customer_date']
    - history_df['order_purchase_timestamp']
).dt.days

avg_delivery_days = history_df.groupby(
    'customer_unique_id'
)['delivery_days'].mean()

print("物流特征完成")


# In[41]:


# ============================================
# 7. 复购标签构造
# ============================================

# 未来窗口中出现过订单的用户
future_users = future_df[
    'customer_unique_id'
].unique()

# 历史用户
all_users = history_df[
    'customer_unique_id'
].unique()

# 标签表
label_df = pd.DataFrame({
    'customer_unique_id': all_users
})

# 是否复购
label_df['repurchase_label'] = label_df[
    'customer_unique_id'
].isin(future_users).astype(int)

print("\n标签分布：")
print(label_df['repurchase_label'].value_counts())


# In[43]:


# ============================================
# 8. 构建最终建模数据集
# ============================================

feature_df = pd.DataFrame()

feature_df['purchase_frequency'] = purchase_frequency
feature_df['avg_order_value'] = avg_order_value
feature_df['total_spending'] = total_spending
feature_df['purchase_interval'] = purchase_interval
feature_df['avg_installments'] = avg_installments
feature_df['favorite_payment'] = favorite_payment
feature_df['favorite_category'] = favorite_category
feature_df['avg_delivery_days'] = avg_delivery_days

# 使用无泄露标签
labels = label_df.set_index(
    'customer_unique_id'
)['repurchase_label']

feature_df['repurchase_label'] = labels

# 重置索引
feature_df.reset_index(inplace=True)

# 删除缺失值
feature_df.dropna(inplace=True)

print("最终建模数据集完成")

print(feature_df.shape)

feature_df.head()


# In[45]:


# ============================================
# 9. 保存处理后数据
# ============================================

feature_df.to_csv(
    'final_model_dataset.csv',
    index=False
)

fact_orders.to_csv(
    'fact_orders.csv',
    index=False
)

dim_customer.to_csv(
    'dim_customer.csv',
    index=False
)

dim_product.to_csv(
    'dim_product.csv',
    index=False
)

dim_payment.to_csv(
    'dim_payment.csv',
    index=False
)

dim_time.to_csv(
    'dim_time.csv',
    index=False
)

print("所有数据保存完成")


# In[47]:


# ============================================
# 10. 数据概览
# ============================================

print("\n========== 数据概览 ==========")

print("最终建模数据:")
print(feature_df.info())

print("\n复购标签分布:")
print(
    feature_df['repurchase_label']
    .value_counts()
)

print("\n描述性统计:")
print(
    feature_df.describe()
)

print("\n处理完成，可进入建模阶段")


# In[ ]:




