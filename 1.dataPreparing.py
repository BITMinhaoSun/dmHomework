import os
import pyarrow.parquet as pq
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os
import json
from sklearn.cluster import KMeans
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import random
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
#  这个部分用于matplotlib的中文
matplotlib_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
if os.path.exists(matplotlib_cache_dir):
    shutil.rmtree(matplotlib_cache_dir)
matplotlib.font_manager.fontManager.addfont('simhei.ttf')

# 设置 Matplotlib 使用 SimHei 字体
matplotlib.rc('font', family='SimHei')
# 配置日志
def configure_logging():
    logging.basicConfig(
        filename='/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/parquet_read.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
# 设置 pandas 显示选项，不以科学计数法显示数字
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 这个函数用于读取目标路径的Parquet 文件
def read_and_display_parquet_files(folder_path):
    all_data = pd.DataFrame()
    # 遍历指定文件夹下的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                try:
                    # 读取 Parquet 文件
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    all_data = pd.concat([all_data, df], ignore_index=True)
                    # 记录文件基本信息到日志
                    # logging.info(f"文件路径: {file_path}")
                    # logging.info("数据基本信息:")
                    # logging.info(str(table.schema))
                    # logging.info(f"数据行数: {table.num_rows}")
                    # logging.info(f"数据列数: {table.num_columns}")
                    #logging.info("数据前几行示例:")
                    #first_few_rows = df[:5]
                    #logging.info(first_few_rows.to_csv(sep='\t', na_rep='nan'))
                    #logging.info("-" * 80)
                except Exception as e:
                    logging.error(f"读取文件 {file_path} 时出现错误: {e}")
    return all_data


# 探索性分析
def exploratory_analysis(data):
    # 查看数据基本信息
    logging.info(data.info())
    # 查看描述性统计信息
    logging.info(data.describe())


# 可视化
def visualization(data):

    # 方式一：柱状图，查看不同性别数量分布
    plt.figure(figsize=(10, 6))
    gender_counts = data['gender'].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/gender_distribution.png')
    plt.close()

    # 方式二：箱线图，查看年龄分布情况
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data['age'])
    plt.title('Age Distribution')
    plt.ylabel('Age')
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/age_distribution.png')
    plt.close()

    # 方式三：散点图，查看年龄和收入的关系
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='age', y='income', data=data)
    plt.title('Scatter Plot of Age and Income')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/age_income_scatter.png')
    plt.close()
    
    # 方式四：直方图，查看收入的分布情况
    plt.figure(figsize=(10, 6))
    sns.histplot(data['income'], bins=10)
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/income_distribution.png')
    plt.close()

    # 方式五：饼图，查看不同国家的用户比例
    country_counts = data['country'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%')
    plt.title('Country Distribution')
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/country_distribution.png')
    plt.close()
    
    # 方式六：热力图，查看各数值型变量之间的相关性
    numerical_data = data.select_dtypes(include=['number'])
    corr_matrix = numerical_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('数值型变量相关性热力图')
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/correlation_heatmap.png')
    plt.close()

    # 方式七：折线图，查看不同年龄段的平均收入趋势
    age_bins = pd.cut(data['age'], bins=10)
    age_income = data.groupby(age_bins)['income'].mean()
    plt.figure(figsize=(10, 6))
    age_income.plot(kind='line')
    plt.title('不同年龄段的平均收入趋势')
    plt.xlabel('年龄段')
    plt.ylabel('平均收入')
    plt.xticks(rotation=45)
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/age_income_trend.png')
    plt.close()

    # 方式八：小提琴图，对比不同性别的年龄分布
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='gender', y='age', data=data)
    plt.title('不同性别的年龄分布')
    plt.xlabel('性别')
    plt.ylabel('年龄')
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/gender_age_violin.png')
    plt.close()

    # 方式九：柱状堆积图，查看不同国家中不同性别的用户数量
    country_gender = data.groupby(['country', 'gender']).size().unstack()
    plt.figure(figsize=(10, 6))
    country_gender.plot(kind='bar', stacked=True)
    plt.title('不同国家中不同性别的用户数量')
    plt.xlabel('国家')
    plt.ylabel('用户数量')
    plt.xticks(rotation=45)
    plt.savefig('/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization/country_gender_stacked_bar.png')
    plt.close()


def evaluate_data_quality(data):
    issues = []
    # 检查缺失值
    missing_values = data.isnull().sum()
    if (missing_values > 0).any():
        issues.append(f"数据中存在缺失值，各列缺失值数量: {missing_values}")
    else:
        logging.info("数据中不存在缺失值")
    # 最终返回一个 Series 对象
    
    # 检查异常值
    age_outliers = data[(data['age'] < 0) | (data['age'] > 150)]
    if not age_outliers.empty:
        issues.append(f"数据中年龄列存在异常值，异常值数据数量: {len(age_outliers)}")
    else:
        logging.info("数据中不存在年龄非法值")    
        
    valid_genders = ['男', '女']
    gender_outliers = data[~data['gender'].isin(valid_genders)]
    if not gender_outliers.empty:
        issues.append(f"数据中性别列存在异常值，异常值数量: {len(age_outliers)}")
    else:
        logging.info("数据中不存在性别非法值")
    # 检查重复数据
    duplicated_rows = data[data.duplicated()]
    if not duplicated_rows.empty:
        issues.append(f"数据中存在重复行，重复行重复量: {len(duplicated_rows)}")

    return issues


# 缺失值处理
def handle_missing_values(data):
    #对于数值型列用均值填充，对于非数值型列用众数填充
    numerical_cols = data.select_dtypes(include=['number']).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

    # non_numerical_cols = data.select_dtypes(exclude=['number']).columns
    # data[non_numerical_cols] = data[non_numerical_cols].fillna('unknown')
    non_numerical_cols = data.select_dtypes(exclude=['number']).columns
    for col in non_numerical_cols:
        mode_value = data[col].mode()[0]
        # .mode()：这是 pandas 中的一个方法，用于计算该列的众数。
        # 由于一列可能存在多个众数（例如，某列中 'A' 和 'B' 出现的次数相同且都是最多的），mode() 方法会返回一个 pandas.Series 对象，其中包含了所有的众数。
        data[col] = data[col].fillna(mode_value)
    # 众数

    return data


# 异常值处理
def handle_outliers(data):
    valid_age_data = data[(data['age'] >= 0) & (data['age'] <= 150)]['age']
    mean_age = valid_age_data.mean()
    # 用平均值填充年龄列的异常值
    data.loc[(data['age'] < 0) | (data['age'] > 150), 'age'] = mean_age
    
    # 对于性别
    valid_genders = ['男', '女']
    data.loc[~data['gender'].isin(valid_genders), 'gender'] = [random.choice(valid_genders) for _ in range(len(data[~data['gender'].isin(valid_genders)]))]
    
    return data


# 重复值处理
def handle_duplicated_rows(data):
    data = data.drop_duplicates()
    remaining_duplicated_mask = data.duplicated(keep=False)
    remaining_duplicated_rows = data[remaining_duplicated_mask]
    if remaining_duplicated_rows.empty:
        logging.info("删除重复行后，数据框中已无重复行。")
    return data

def cluster_and_analyze(data):
    """
    对数据进行指定列的提取、处理，然后进行K-means聚类，并分析每个聚类的特征。
    :param data: 包含用户数据的DataFrame
    :return: 每个聚类的特征分析结果（DataFrame）
    """
    logging.info("开始执行 cluster_and_analyze 函数")
    save_path = '/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization'
    def parse_purchase_history(x):
        try:
            parsed = json.loads(x.replace('""', '"'))
            return parsed['average_price'], len(parsed['items'])
        except (KeyError, json.JSONDecodeError):
            logging.warning(f"解析 purchase_history 时出现错误: {x}")
            return None, None

    logging.info("开始提取 purchase_history 中的 average_price 和 items 的长度")
    data['average_price'], data['items_count'] = zip(*data['purchase_history'].apply(parse_purchase_history))
    data = data.dropna(subset=['average_price', 'items_count'])  # 去除解析失败的数据
    logging.info("提取完成")

    # 选择特征
    features = data[['income', 'average_price', 'items_count']]
    logging.info(f"选择的特征为: {features.columns.tolist()}")

    # 数据标准化
    logging.info("开始进行数据标准化")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    logging.info("数据标准化完成")

    ########################这个部分
    # sse = []
    # 
    # # 确定最优聚类数
    # # 经过预先观察，8比较合适
    logging.info("开始确定最优聚类数")
    silhouette_scores = []
    k_range = range(1, 25)
    for k in k_range:
        logging.info(f"开始尝试聚类数 k = {k}")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        sse.append(kmeans.inertia_)
        logging.info(f"完成聚类数 k = {k} 的尝试，当前 SSE: {kmeans.inertia_}")

    # 绘制 SSE 随聚类数变化的曲线
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared errors (SSE)')
    plt.title('Elbow Method For Optimal k')
    elbow_file_path = os.path.join(save_path, 'Elbow_cluster_visualization.png')
    plt.savefig(elbow_file_path)
    ##############
    logging.info(f"最优聚类数 k = {best_k}")
    ##############
    # 进行 K-means 聚类
    logging.info(f"开始使用最优聚类数 k = {best_k} 进行 K-means 聚类")
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    data['cluster'] = kmeans.fit_predict(features_scaled)
    logging.info("K-means 聚类完成")

    # 分析每个聚类的特征
    logging.info("开始分析每个聚类的特征")
    cluster_analysis = data.groupby('cluster').agg({
        'income': 'mean',
        'average_price': 'mean',
        'items_count': 'mean',
    })
    logging.info("聚类特征分析完成")
    # 添加 PCA 降维步骤
    logging.info("开始进行 PCA 降维以可视化聚类结果")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    data['pca_1'] = features_pca[:, 0]
    data['pca_2'] = features_pca[:, 1]
    logging.info("PCA 降维完成")
    logging.info("开始绘制散点图可视化聚类结果")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x='pca_1', y='pca_2', hue='cluster', palette='viridis', s=50)
    plt.title('K-means Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()

    # 保存图片到指定路径
    save_path = '/home/sunminhao/grade8_HOMEWORK/DataMining/Homework1/visualization'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, 'cluster_visualization.png')
    logging.info(f"将可视化结果保存到 {file_path}")
    plt.savefig(file_path)
    plt.show()
    logging.info("可视化结果保存完成")

    logging.info("cluster_and_analyze 函数执行结束")
    return cluster_analysis

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif']=['SimHei']
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font='SimHei')
    configure_logging()
    # 文件路径
    # folder_path = '/home/data/public/dataMiningCourseData/1G_data/'
    # folder_path = '/home/data/public/dataMiningCourseData/10G_data/10G_data'
    folder_path = '/home/data/public/dataMiningCourseData/30G_data/30G_data'
    # 读取数据
    combined_data = read_and_display_parquet_files(folder_path)
    # 简单展示一些内容
    exploratory_analysis(combined_data)
    # 数据质量评价
    data_quality_issues = evaluate_data_quality(combined_data)
    for issue in data_quality_issues:
        logging.warning(issue)

    # 数据预处理
    preprocessed_data = combined_data
    preprocessed_data = handle_missing_values(preprocessed_data)
    preprocessed_data = handle_outliers(preprocessed_data)
    preprocessed_data = handle_duplicated_rows(preprocessed_data)
    # 可视化
    visualization(preprocessed_data)

    cluster_result = cluster_and_analyze(preprocessed_data)
    logging.info("每个聚类的特征分析：")
    logging.info(cluster_result)

    