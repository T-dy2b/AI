import urllib.request as request
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta, timezone

url = 'https://info.finance.yahoo.co.jp/fx/detail/?code=usdjpy'
url2 = 'https://www.nikkei.com/markets/worldidx/chart/nk225/'
url3 = 'https://tenki.jp/live/5/26/'

response = request.urlopen(url)
response2 = request.urlopen(url2)
response3 = request.urlopen(url3)

bs = BeautifulSoup(response, 'html.parser')
bs2 = BeautifulSoup(response2, 'html.parser')
bs3 = BeautifulSoup(response3, 'html.parser')

stocksPrice = bs.select('#USDJPY_detail_bid')[0].text
print(stocksPrice)

aksPrice = bs.select('#USDJPY_detail_ask')[0].text
print(aksPrice)

kabuPrice = bs2.select('.economic_value_now')[0].text
print(kabuPrice)

aititenki = bs3.select('.weather_entry_telop')[0].text
print(aititenki)

aitikion = bs3.select('.temp-entry')[0].text
print(aitikion)

jst = timezone(timedelta(hours=+9), 'JST')
today = datetime.now(jst)
print(today)

df = pd.DataFrame(data = [[today, stocksPrice,aksPrice,kabuPrice,aititenki,aitikion]])
df.to_csv('fx_price.csv', mode = 'a', header = False, index = False)

