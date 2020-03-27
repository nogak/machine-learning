import pandas as np

# read data_flame
test_df = pd.read_csv('hoge.csv', sep=',') # sep : 区切り文字

# 最初の数行確認
test_df.head()

# 行，列数の確認
print(test_df.shape)
# データタイプの確認
print(test_df.dtype)

# 欠損値の確認
print(test_df.isnull().any(axis=1))
print(test_df.isnull().any(axis=0))
# 欠損値の個数の確認
print(test_df.isnull().sum(axis=1))
print(test_df.isnull().sum(axis=0))

# 統計量の計算
test_df.describe()

# データの可視化-----
import matplotlib.pyplot as plt
%matplotlib inline

plt.hist(test_df['age'])
plt.xlabel('age')
plt.ylabel('freq')
plt.show()
# -------------------

# 散布図のプロット------
plt.scatter(test_df['age'], test_df['balance'])
plt.xlabel('age')
plt.ylabel('balance')
plt.show()

test_df[['age', 'balance']].corr() # 相関係数
# ----------------------

# 値ラベルと値の出現数(比率)を確認
print(test_df.['job'].value_counts(ascending=False, normalize=True))

# 円グラフの作成-----
job_label = test_df['job'].value_counts(ascending=False, normalize=True).index
job_vals = test_df['job'].value_counts(ascending=False, normalize=True).values

plt.pie(job_vals, labels=job_label)
plt.axis('equal')
plt.show()
# -------------------

# 箱ひげ図の作成-----
y_yes = test_df['y' == 'yes']
y_no = test_df['y' == 'no']
y_age = [y_yes['age'], y_no['age']]

plt.boxplot(y_age)
plt.xlabel('y')
plt.ylabel('age')
ax = plt.gca()
plt.setp(ax, xticklabels = ['yes', 'no'])
plt.show()
# -------------------

# 欠損値の除外
test_df = test_df.dropna(subset=['job', 'education'])
# 欠損値の補完
test_df = test_df.fillna({'contact': 'unknown'})
# 外れ値の除去
test_df = test_df[test_df['age'] >= 18]
test_df = test_df[test_df['age'] < 100]

# 文字列を数値へ変換
test_df = test_df.replace('yes', 1)
test_df = test_df.replace('no', 0)

# 多値ラベルをOne-hot表現でダミー変数化
test_df_job = pd.get_dummies(test_df['job'])