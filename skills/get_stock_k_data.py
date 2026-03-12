import requests
import pandas as pd
import time
import random
import sys
from datetime import datetime

def get_stock_k_data(international_code, start_date='2023-01-01', end_date='2025-5-16', klt=101, max_retries=3):
    """
    获取股票K线数据
    klt: 1=1分钟, 5=5分钟, 15=15分钟, 30=30分钟, 60=60分钟, 101=日K, 102=周K, 103=月K
    max_retries: 最大重试次数
    """
    symbol = international_code.split('.')[0]
    if international_code.endswith('.XSHG'):
        eastmoney_prefix = '1'  # 东方财富 1 开头为上交所
    elif international_code.endswith('.XSHE'):
        eastmoney_prefix = '0'  # 东方财富 0 开头为深交所
    elif international_code.endswith('.HK'):
        eastmoney_prefix = '116'  # 东方财富 0 开头为深交所
    elif international_code.endswith('.US'):
        eastmoney_prefix = '105'  # 东方财富 0 开头为深交所
    else:
        raise ValueError('市场类型错误，应为 "XSHE" 或 "XSHG"')
    url = f'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': f"{eastmoney_prefix}.{symbol}",
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': klt,  # K线类型
        'fqt': 1,    # 前复权
        'beg': start_date.replace('-', ''),
        'end': end_date.replace('-', ''),
        'lmt': 10000,
    }

    # 随机User-Agent列表
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]

    # 重试机制
    for retry in range(max_retries):
        try:
            headers = {
                'User-Agent': random.choice(user_agents)
            }
            r = requests.get(url, params=params, headers=headers, timeout=30)
            data = r.json()
            if not data or 'data' not in data or not data['data'] or 'klines' not in data['data']:
                print('接口返回异常，原始响应如下:')
                print(data)
                if retry == max_retries - 1:
                    raise ValueError('未获取到有效K线数据，请检查股票代码、市场参数或稍后重试。')
                wait_time = random.uniform(2.0, 5.0)
                print(f"等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
                continue
            
            kline = data['data']['klines']
            df = pd.DataFrame([i.split(',') for i in kline],
                        columns=['date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'pct_change', 'change', 'turnover_rate'])
            return df
            
        except Exception as e:
            print(f"请求失败 ({retry+1}/{max_retries}): {e}")
            if retry == max_retries - 1:  # 最后一次重试
                raise
            # 随机等待时间，避免被限流
            wait_time = random.uniform(3.0, 10.0)
            print(f"等待 {wait_time:.2f} 秒后重试...")
            time.sleep(wait_time)
    
    # 如果所有重试都失败了
    raise ValueError(f"抓取股票 {international_code} K线数据失败")


if __name__ == '__main__':
    
    data = get_stock_k_data('TSLA.US', start_date='2026-3-11', end_date='2026-3-11',klt=1)
    print(data)