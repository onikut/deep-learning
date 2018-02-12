import time
import poloniex
import pandas as pd

# poloniex APIの準備
polo = poloniex.Poloniex()

# 5分間隔（サンプリング間隔300秒）で100日分読み込む
chart_data = polo.returnChartData('BTC_ETH', period=300, start=time.time()-polo.DAY*100, end=time.time())
#print(chart_data)

# pandasにデータの取り込み
df = pd.DataFrame(chart_data)
df.head(10)


# 短期線：窓幅1日（5分×12×24）
data_s = pd.rolling_mean(df['close'], 12 * 24)

# 長期線：窓幅5日（5分×12×24×5）
data_l = pd.rolling_mean(df['close'], 12 * 24 * 5)

# matplotlibの読み込み（エラーが出た時はpip or pip3でインストール）
import matplotlib.pyplot as plt

# 一番簡単なプロット
plt.plot(df['close'])
plt.show()

# 描画を綺麗に表示する
from matplotlib.pylab import rcParams
import seaborn as sns
rcParams['figure.figsize'] = 15, 6

# プロットの色を指定しよう（color）
plt.plot(df['close'], color='#7f8c8d')
plt.show()
