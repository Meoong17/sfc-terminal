#!/usr/bin/env python3
"""Fetch CME BTC1! continuous futures daily history from TradingView (anon).
Not part of SFC scoring — reference/institutional price source check."""
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import time, os

def get_tradingview_btc1():
    print("Menghubungkan ke TradingView secara anonim...")
    tv = TvDatafeed()
    print("Menarik BTC1! (CME Continuous Futures)...")
    try:
        btc1_data = tv.get_hist(symbol='BTC1!', exchange='CME', interval=Interval.in_daily, n_bars=5000)
        if btc1_data is not None and len(btc1_data) > 0:
            btc1_data.index.name = 'datetime'
            btc1_data.reset_index(inplace=True)
            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tradingview_btc1_daily.csv")
            btc1_data.to_csv(filename, index=False)
            print(f"Data berhasil: {len(btc1_data)} baris -> {filename}")
            print("Rentang:", btc1_data['datetime'].min(), "..", btc1_data['datetime'].max())
            print(btc1_data.tail(3))
            return filename
        else:
            print("Gagal/kosong: tidak ada data dikembalikan.")
    except Exception as e:
        print(f"Error: {e}")
    return None

if __name__ == "__main__":
    get_tradingview_btc1()
