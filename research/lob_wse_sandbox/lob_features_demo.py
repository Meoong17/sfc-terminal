#!/usr/bin/env python3
"""
lob_features_demo.py — WSELOB-2017 methodology sandbox (NOT part of SFC scoring).
Demonstrates the full LOB methodology on a slice of the raw WSE message stream:
  1. load one day's LOB messages (HDF5 record array)
  2. reconstruct the limit order book with the author's OrderBook class
  3. compute canonical LOB features over time: mid, spread, top-5 depth,
     order-book imbalance (OBI), weighted-average price (WAP)
  4. write a small feature sample

The feature concepts here (mid, spread, depth, OBI, flow imbalance) transfer
directly to crypto limit-order-book data if/when SFC obtains BTC tick/LOB data.
"""
import os, sys
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orderbook2 import OrderBook

HERE = os.path.dirname(os.path.abspath(__file__))
LOB = os.path.join(HERE, "PEKAO_lob_2017_zlib.h5")

def load_day(day="d20170102", nlimit=12000):
    """Load first `nlimit` messages of one day as a structured array."""
    with h5py.File(LOB, "r") as f:
        tbl = f[day]["table"]
        return tbl[0:nlimit]

def msg_to_order(m):
    """Map one raw message to the OrderBook.add_order/mod/del API signature."""
    return {
        "order_date": int(m["order_date"]),
        "priority_date": int(m["priority_date"]),
        "order_id": int(m["order_id"]),
        "price": float(m["price"]) / 100.0,   # WSE price tick /100 -> PLN
        "volume": int(m["volume"]),
        "order_type": m["order_type"].decode(),
    }

def process(msgs, snapshot_every=2000):
    ob = OrderBook()
    rows = []
    action_counts = {}
    for i, m in enumerate(msgs):
        at = m["action_type"].decode()
        action_counts[at] = action_counts.get(at, 0) + 1
        order = msg_to_order(m)
        side = int(m["side"])
        if at == "A":
            o = dict(order, side=side); ob.add_order(o)
        elif at == "M":
            o = dict(order, side=side); ob.mod_order(o)
        elif at == "D":
            o = dict(order, side=side); ob.del_order(o)
        elif at == "Y":
            o = dict(order, side=side); ob.retransmit_order(o)
        elif at == "F":
            ob.clear_orderbook()
        if (i + 1) % snapshot_every == 0 and not ob.isempty():
            try:
                bests = ob.get_bests(num=5, cum=True)
                bb = bests["best_buy_prices"]; bs = bests["best_sell_prices"]
                bv = bests["best_buy_volumes"]; sv = bests["best_sell_volumes"]
                mid = ob.get_mid(); sp = ob.get_spread()
                bid_d = bv.sum(); ask_d = sv.sum()
                obi = (bid_d - ask_d) / (bid_d + ask_d) if (bid_d + ask_d) > 0 else np.nan
                rows.append({
                    "msg": i + 1, "best_bid": bb[0], "best_ask": bs[0],
                    "mid": mid, "spread": sp,
                    "depth_bid_5": bid_d, "depth_ask_5": ask_d,
                    "obi_5": obi, "wap": ob.get_wap(level=5, mode="combine"),
                })
            except Exception:
                pass
    return rows, action_counts

def main():
    msgs = load_day(nlimit=12000)
    print(f"Loaded {len(msgs)} messages from PEKAO 2017-01-02")
    print(f"Message-type counts (action_type): { {k: msgs['action_type'].astype('U1').tolist().count(k) for k in set(msgs['action_type'].astype('U1').tolist())} }")
    rows, counts = process(msgs)
    print(f"Action dispatch counts: {counts}")
    print("\nLOB feature snapshots (every 2000 msgs):")
    if not rows:
        print("  (none produced — slice too small / book never non-empty)")
        return
    hdr = f"{'msg':>6} {'best_bid':>8} {'best_ask':>8} {'mid':>8} {'spread':>6} {'depth5b':>8} {'depth5a':>8} {'OBI5':>7} {'WAP':>8}"
    print(hdr)
    for r in rows:
        print(f"{r['msg']:>6} {r['best_bid']:>8.2f} {r['best_ask']:>8.2f} {r['mid']:>8.2f} "
              f"{r['spread']:>6.2f} {r['depth_bid_5']:>8.0f} {r['depth_ask_5']:>8.0f} "
              f"{r['obi_5']:>7.3f} {r['wap']:>8.2f}")
    print("\nMethodology notes: price scaled /100 (PLN); volume in shares; OBI = (bidV-askV)/(bidV+askV) top-5.")

if __name__ == "__main__":
    main()
