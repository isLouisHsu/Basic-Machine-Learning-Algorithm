import os
import datetime
import pandas as pd
import pandas_datareader.data as web

file_path = './data/yahoo.csv'

def load_data(file_path):
    """
    Date    High	Low	   Open	  Close	    Volume	    Adj Close
    日期    最低价   最低价 开盘价	收盘价	   成交量	   复权收盘价 
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        start = datetime.datetime(2010, 1, 1)
        end = datetime.datetime.now()
        df = web.DataReader('AAPL', "yahoo", start=start, end=end)
        df.to_csv(file_path)
    return df

# 指标计算
def avr():
    """
    股票平均价格 = 成交量 × 成交价 / 总成交股数 × 100%
    """
    pass
def asi():
    """
    1.  A=∣当天最高价-前一天收盘价∣
        B=∣当天最低价-前一天收盘价∣
        C=∣当天最高价-前一天最低价∣
        D=∣前一天收盘价-前一天开盘价∣
    2.  比较A、B、C三数值：
        若A最大，R=A+1/2B+1/4D
        若B最大，R=B+1/2A+1/4D
        若C最大，R=C+1/4D
    3.  E=当天收盘价-前一天收盘价
        F=当天收盘价-当天开盘价
        G=前一天收盘价-前一天开盘价
    4.  X=E+1/2F+G
    5.  K=A、B之间的最大值
    6.  L=3；SI=50*X/R*K/L；ASI=累计每日之SI值
    """
    pass
def rsi():
    """
    RSI:= SMA(MAX(Close-LastClose,0),N,1)/SMA(ABS(Close-LastClose),N,1)*100
    """
    pass
def rsv():
    """
    RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
    以9日为周期的KD线为例。首先须计算出最近9日的RSV值，即未成熟随机值，计算公式为
    9日RSV=（当日的收盘价－9日内的最低价）÷（9日内的最高价－9日内的最低价）×100 （计算出来的数值为当日的RSV）
    """
    pass
def kdj():
    """
    KDJ的计算比较复杂，首先要计算周期（n日、n周等）的RSV值，即未成熟随机指标值，然后再计算K值、D值、J值等。以n日KDJ数值的计算为例，其计算公式为
    n日RSV=（Cn－Ln）/（Hn－Ln）×100
    公式中，Cn为第n日收盘价；Ln为n日内的最低价；Hn为n日内的最高价。
    其次，计算K值与D值：
    当日K值=2/3×前一日K值+1/3×当日RSV
    当日D值=2/3×前一日D值+1/3×当日K值
    若无前一日K 值与D值，则可分别用50来代替。
    J值=3*当日K值-2*当日D值
    以9日为周期的KD线为例，即未成熟随机值，计算公式为
    9日RSV=（C－L9）÷（H9－L9）×100
    公式中，C为第9日的收盘价；L9为9日内的最低价；H9为9日内的最高价。
    K值=2/3×第8日K值+1/3×第9日RSV
    D值=2/3×第8日D值+1/3×第9日K值
    J值=3*第9日K值-2*第9日D值
    若无前一日K值与D值，则可以分别用50代替
    """
    pass
def bias():
    """
    乖离率=[(当日收盘价-N日平均价)/N日平均价]*100%
    BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日
    """
    pass
def ma():
    """
    N日移动平均线=N日收市价之和/N
    以时间的长短划分，移动平均线可分为短期、中期、长期几种，
    短期移动平均线5天与10天；
    中期有30天、65天；
    长期有200天及280天。
    """
    pass
def ema():
    """
    EMAtoday=α * ( Pricetoday - EMAyesterday ) + EMAyesterday;
    """
    pass
def sma():
    """
    SMA(X,N,M)，求X的N日移动平均，M为权重。
    SMA(X,N,M) = (M*X+(N-M)*Y')/N，其中Y'表示上一周期Y值，N必须大于M。
    """
    pass
def bbi():
    """
    BBI=(3日均价+6日均价+12日均价+24日均价)÷4
    """
    pass
if __name__ == '__main__':
    df = load_data(file_path)