import requests
import pandas as pd
import time
import random
from datetime import datetime


def normalize_date(date_str: str) -> str:
    """
    将任意 'YYYY-M-D' 或 'YYYY-MM-DD' 格式日期转为东方财富接口要求的 YYYYMMDD 8位格式
    """
    for fmt in ('%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y%m%d')
        except ValueError:
            continue
    raise ValueError(f"无法解析日期格式: {date_str}，请使用 YYYY-MM-DD 格式")


def normalize_code(code: str) -> str:
    """
    自动识别股票/ETF代码，补全为 international_code 格式
    支持：A股、ETF、港股、美股
    """
    code = code.strip().upper()

    # 已经是完整格式，直接返回
    if any(code.endswith(s) for s in ['.XSHG', '.XSHE', '.HK', '.US']):
        return code

    # 纯数字：A股 或 ETF
    if code.isdigit():
        if code.startswith(('60', '68', '51', '58')):    # 上交所
            return f"{code}.XSHG"
        elif code.startswith(('00', '30', '15', '16')):  # 深交所
            return f"{code}.XSHE"
        elif len(code) == 5:                              # 港股（5位数字）
            return f"{code}.HK"
        else:
            raise ValueError(f"无法识别的数字代码: {code}")

    # 纯字母：美股
    if code.isalpha():
        return f"{code}.US"

    raise ValueError(f"无法识别的代码格式: {code}")


def get_stock_k_data(code, start_date='2023-01-01', end_date='2026-03-14', klt=101, max_retries=3):
    """
    获取股票/ETF K线数据

    参数：
        code        : 股票代码，支持多种格式：
                        A股/ETF: '600519', '000001', '159887', '510300'
                        美股:    'TSLA', 'AAPL', 'CRCL'
                        港股:    '09988', '00700'
                        完整格式: '600519.XSHG', 'TSLA.US' 等
        start_date  : 开始日期，格式 'YYYY-MM-DD'（支持不补零，如 '2025-3-1'）
        end_date    : 结束日期，格式 'YYYY-MM-DD'
        klt         : K线类型
                        1=1分钟, 5=5分钟, 15=15分钟, 30=30分钟,
                        60=60分钟, 101=日K, 102=周K, 103=月K
        max_retries : 最大重试次数
    
    返回：
        pd.DataFrame，列：date, open, close, high, low, volume, turnover,
                          amplitude, pct_change, change, turnover_rate
    """
    international_code = normalize_code(code)
    symbol = international_code.split('.')[0]

    # 确定要尝试的 secid 前缀列表
    if international_code.endswith('.XSHG'):
        prefixes = ['1']
    elif international_code.endswith('.XSHE'):
        prefixes = ['0']
    elif international_code.endswith('.HK'):
        prefixes = ['116']
    elif international_code.endswith('.US'):
        prefixes = ['105', '106', '107']  # 105=NASDAQ, 106=NYSE，自动尝试两个
    else:
        raise ValueError('市场类型错误')

    # 标准化日期格式，确保为 YYYYMMDD 8位
    beg = normalize_date(start_date)
    end = normalize_date(end_date)

    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    ]

    last_error = None

    for prefix in prefixes:
        secid = f"{prefix}.{symbol}"
        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': klt,
            'fqt': 1,      # 前复权
            'beg': beg,
            'end': end,
            'lmt': 10000,
        }

        for retry in range(max_retries):
            try:
                headers = {'User-Agent': random.choice(user_agents)}
                r = requests.get(url, params=params, headers=headers, timeout=30)
                data = r.json()

                if (data and 'data' in data and data['data']
                        and 'klines' in data['data'] and data['data']['klines']):
                    kline = data['data']['klines']
                    df = pd.DataFrame(
                        [i.split(',') for i in kline],
                        columns=['date', 'open', 'close', 'high', 'low', 'volume',
                                 'turnover', 'amplitude', 'pct_change', 'change', 'turnover_rate']
                    )
                    print(f"成功获取数据，secid={secid}，共 {len(df)} 条记录")
                    return df

                # 当前前缀无数据，跳出重试循环，尝试下一个前缀
                print(f"secid={secid} 未返回有效数据，尝试下一个前缀...")
                break

            except Exception as e:
                last_error = e
                print(f"请求失败 secid={secid} ({retry + 1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    wait_time = random.uniform(3.0, 10.0)
                    print(f"等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)

    raise ValueError(
        f"抓取 [{code}] K线数据失败，已尝试前缀 {prefixes}，"
        f"请检查：1)代码是否正确 2)上市日期是否在查询区间内 3)东方财富是否收录该标的\n"
        f"最后错误: {last_error}"
    )


if __name__ == '__main__':
    test_cases = [
        # (代码,         开始日期,       说明)
        ('TSLA',        '2025-01-01',   '美股 NASDAQ'),
        ('CRCL',        '2025-06-05',   '美股 NYSE，2025年6月上市'),
        ('APLX',        '2025-09-08',   '美股杠杆ETF，2025年9月上市'),
        ('BABA',      '2025-01-01',   'A股 上交所'),
        ('000001',      '2025-01-01',   'A股 深交所'),
        ('159887',      '2025-01-01',   'ETF 深交所'),
        ('510300',      '2025-01-01',   'ETF 上交所'),
    ]

    end = '2026-03-14'

    for code, start, desc in test_cases:
        print(f"\n{'='*50}")
        print(f"测试: {code} ({desc})")
        try:
            df = get_stock_k_data(code, start_date=start, end_date=end, klt=101)
            print(df.tail(3))
            print(type(df))
        except ValueError as e:
            print(f"[跳过] {e}")