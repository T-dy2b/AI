import urllib.request as request
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta, timezone

url = 'https://www.nikkei.com/markets/worldidx/chart/nk225/'

response = request.urlopen(url)

bs = BeautifulSoup(response, 'html.parser')

stocksPrice = bs.select('.economic_value_now')[0].text
print(stocksPrice)

# 今日の日付を取得
jst = timezone(timedelta(hours=+9), 'JST')
today = datetime.now(jst).date().isoformat()

# PandasのDataFrameを生成
df = pd.DataFrame(data = [[today, stocksPrice]])

# CSVに保存（追記）
df.to_csv('stocks_price.csv', mode = 'a', header = False, index = False)


