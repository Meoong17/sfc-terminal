from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class OrderBook(object):
    def __init__(self, data=None, verbose=False):
        super(OrderBook, self).__init__()
        if data is None:
            self.current_time = None
            self.nmsg = 0
            self.buy_orders = {'order_date': [], 'priority_date': [], 'order_id': [], 'price': [], 'volume': [],
                               'order_type': []}
            self.sell_orders = {'order_date': [], 'priority_date': [], 'order_id': [], 'price': [], 'volume': [],
                                'order_type': []}
        else:
            self.nmsg = deepcopy(data.get('nmsg', 0))
            self.current_time = deepcopy(data['current_time'])
            self.buy_orders = deepcopy(data['buy_orders'])
            self.sell_orders = deepcopy(data['sell_orders'])
        self.verbose = verbose

    def copy(self):
        return deepcopy(self)

    def isempty(self):
        if len(self.buy_orders['order_date']) == 0 or len(self.sell_orders['order_date']) == 0:
            return True
        return False
    
    def isthesame(self, ob2):
        l = self.buy_orders['volume']
        r = ob2.buy_orders['volume']
        if (l.shape != r.shape) or (l != r).any():
            return False
        l = self.sell_orders['volume']
        r = ob2.sell_orders['volume']
        if (l.shape != r.shape) or (l != r).any():
            return False
        l = self.buy_orders['price']
        r = ob2.buy_orders['price']
        if (l.shape != r.shape) or (l != r).any():
            return False
        l = self.sell_orders['price']
        r = ob2.sell_orders['price']
        if (l.shape != r.shape) or (l != r).any():
            return False
        return True
    
    def to_min_dict(self):
        d = {'current_time': deepcopy(self.current_time), 'nmsg': deepcopy(self.nmsg)}
        bests = self.get_bests(num=None, cum=True)
        d['buy_prices'] =  deepcopy(bests['best_buy_prices'])
        d['sell_prices'] =  deepcopy(bests['best_sell_prices'])
        d['buy_volumes'] =  deepcopy(bests['best_buy_volumes'])
        d['sell_volumes'] =  deepcopy(bests['best_sell_volumes'])
        return d

    def to_dict(self):
        d = {'current_time': deepcopy(self.current_time), 'nmsg': deepcopy(self.nmsg), 'buy_orders': deepcopy(self.buy_orders),
             'sell_orders': deepcopy(self.sell_orders)}
        return d

    def reduce_to_nlevels(self, nlevels=10):
        ob = self.copy()
        if isinstance(nlevels, int):
            bests = self.get_bests(num=nlevels, cum=True)
            price_up = bests['best_sell_prices'][-1]
            price_down = bests['best_buy_prices'][-1]
        else:
            mid = self.get_mid()
            price_up = (1+nlevels)*mid
            price_down = (1-nlevels)*mid
        ob = ob.remove_orders_by_price(price_up, 'sell')
        ob = ob.remove_orders_by_price(price_down, 'buy')
        return ob

    def display(self, cum=True, out=False):
        dbuys = self.get_buys_df(cum=cum)
        dsells = self.get_sells_df(cum=cum)
        if cum:
            dbuys = dbuys.reset_index().droplevel(1, axis=1)
            dbuys.columns = ['price', 'order_type', 'volume', 'size']
            dsells = dsells.reset_index().droplevel(1, axis=1)
            dsells.columns = ['price', 'order_type', 'volume', 'size']
        dbuys = dbuys[['price', 'volume']]
        dsells = dsells[['price', 'volume']]
        tab = pd.merge(dbuys, dsells, how='outer', on='price', suffixes=('_bid', '_ask')).reindex(
            columns=['volume_bid', 'price', 'volume_ask']).sort_values(by='price', ascending=False)
        width = max(10, len(str(tab.max().max())) + 3)
        stab = tab.to_string(index=False, na_rep='', justify='center', formatters=(
            lambda x: '{:.0f}'.format(x).rjust(width), lambda x: '{:.2f}'.format(x).center(width),
            lambda x: '{:.0f}'.format(x).ljust(width)))
        stab = 'Current time is {}\n'.format(self.current_time) + stab
        if out:
            return stab
        else:
            print(stab)

    def summary(self, out=False):
        dbuys = self.get_buys_df(cum=False)
        dsells = self.get_sells_df(cum=False)
        lask = len(dsells['price'].unique())
        lbid = len(dbuys['price'].unique())
        oask = len(dsells)
        obid = len(dbuys)
        vask = dsells['volume'].sum()
        vbid = dbuys['volume'].sum()

        bests = self.get_bests()
        bpask = bests['best_sell_prices'][0]
        bpbid = bests['best_buy_prices'][0]
        bvask = bests['best_sell_volumes'][0]
        bvbid = bests['best_buy_volumes'][0]

        mid = 0.5 * (bpask + bpbid)
        spread = bpask - bpbid
        wap = self.get_wap()

        s = """Current time is {}

Ask price levels:   {}
Bid price levels:   {}
Total price levels: {}

Ask orders:         {}
Bid orders:         {}
Total orders:       {}

Ask volume:         {}
Bid volume:         {}
Total volume:       {} 

Spread:             {:.2f}
Mid point:          {:.2f}
WAP:                {:.2f}

Best Ask:           {:.2f}
Volume:             {}

Best Bid:           {:.2f}
Volume:             {}
        """.format(self.current_time, lask, lbid, lask+lbid, oask, obid, oask+obid, vask,
                   vbid, vask+vbid, spread, mid, wap, bpask, bvask, bpbid, bvbid)
        if out:
            return s
        else:
            print(s)

    def plot(self, cum=True, bounds=None, ref=None, ax=None, figsize=(10, 10), height=0.1):
        dbuys = self.get_buys_df(cum=cum)
        dsells = self.get_sells_df(cum=cum)
        if cum:
            dbuys = dbuys.reset_index().droplevel(1, axis=1)
            dbuys.columns = ['price', 'order_type', 'volume', 'size']
            dsells = dsells.reset_index().droplevel(1, axis=1)
            dsells.columns = ['price', 'order_type', 'volume', 'size']
        dbuys = dbuys[['price', 'volume']]
        dsells = dsells[['price', 'volume']]
        if ref is None:
            ref = self.get_mid()
        if bounds is not None:
            dbuys = dbuys[(ref*(1-bounds) <= dbuys['price']) & (dbuys['price'] <= ref*(1+bounds))]
            dsells = dsells[(ref * (1 - bounds) <= dsells['price']) & (dsells['price'] <= ref * (1 + bounds))]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.barh(dsells['price'].values, dsells['volume'].values, align='center', height=height)
        ax.barh(dbuys['price'].values, -dbuys['volume'].values, align='center', height=height)
        ax.set_title('Order Book')
        ax.set_title('BID', loc='left')
        ax.set_title('ASK', loc='right')
        a = ax.get_xticks().tolist()
        ax.axvline(x=0, color='k', linewidth=1)
        ax.xaxis.set_major_locator(mticker.FixedLocator(a))
        for i in range(len(a)):
            a[i] = int(abs(a[i]))
        ax.set_xticklabels(a)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        if ax is None:
            plt.show()

    def plot_sd(self, bounds=None, ref=None, ax=None, figsize=(5, 10)):
        dbuys = self.get_buys_df(cum=True)
        dsells = self.get_sells_df(cum=True)
        dbuys = dbuys.reset_index().droplevel(1, axis=1)
        dbuys.columns = ['price', 'order_type', 'volume', 'size']
        dsells = dsells.reset_index().droplevel(1, axis=1)
        dsells.columns = ['price', 'order_type', 'volume', 'size']
        dbuys = dbuys[['price', 'volume']]
        dsells = dsells[['price', 'volume']]
        dbuys['cum_volume'] = dbuys['volume'].cumsum()
        dsells['cum_volume'] = dsells['volume'].cumsum()
        if ref is None:
            ref = self.get_mid()
        if bounds is not None:
            dbuys = dbuys[(ref*(1-bounds) <= dbuys['price']) & (dbuys['price'] <= ref*(1+bounds))]
            dsells = dsells[(ref * (1 - bounds) <= dsells['price']) & (dsells['price'] <= ref * (1 + bounds))]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.step(dsells['cum_volume'], dsells['price'], where='pre', label='Supply')
        ax.step(dbuys['cum_volume'], dbuys['price'], where='pre', label='Demand')
        ax.set_title('Supply and Demand')
        ax.legend()
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        if ax is None:
            plt.show()

    def add_perturbation(self, perc=0.05, rl=10, tick_size=0.01):
        ob = self.copy()
        bests_10l = ob.get_bests(num=rl, cum=True)
        sell_p = bests_10l['best_sell_prices']
        buy_p = bests_10l['best_buy_prices']
        smin, smax = np.min(sell_p), np.max(sell_p)
        bmin, bmax = np.min(buy_p), np.max(buy_p)
        srt = np.arange(smin, smax+tick_size, tick_size)
        brt = np.arange(bmin, bmax+tick_size, tick_size)
        s = np.array(list(set((srt*100).astype(int))-set((sell_p*100).astype(int))))
        b = np.array(list(set((brt*100).astype(int))-set((buy_p*100).astype(int))))
        r_mask = np.random.random(len(s))<perc
        s = s[r_mask]
        r_mask = np.random.random(len(b))<perc
        b = b[r_mask]

        i = 0
        for p in s:
            i -= 1
            order = {'order_date': pd.Timestamp.now(tz='Europe/Warsaw'),
                     'priority_date': pd.Timestamp.now(tz='Europe/Warsaw'),
                     'order_id': i,
                     'price': p/100,
                     'volume': 1,
                     'order_type': 'p',
                     'side': 2}
            ob.add_order(order)
        for p in b:
            i -= 1
            order = {'order_date': pd.Timestamp.now(tz='Europe/Warsaw'),
                     'priority_date': pd.Timestamp.now(tz='Europe/Warsaw'),
                     'order_id': i,
                     'price': p/100,
                     'volume': 1,
                     'order_type': 'p',
                     'side': 1}
            ob.add_order(order)
        return ob

    def add_order(self, order):  # A
        if order['side'] == 1:
            self.buy_orders['order_date'].append(order['order_date'])
            self.buy_orders['priority_date'].append(order['priority_date'])
            self.buy_orders['order_id'].append(order['order_id'])
            self.buy_orders['price'].append(order['price'])
            self.buy_orders['volume'].append(order['volume'])
            self.buy_orders['order_type'].append(order['order_type'])
        elif order['side'] in [2, 5]:
            self.sell_orders['order_date'].append(order['order_date'])
            self.sell_orders['priority_date'].append(order['priority_date'])
            self.sell_orders['order_id'].append(order['order_id'])
            self.sell_orders['price'].append(order['price'])
            self.sell_orders['volume'].append(order['volume'])
            self.sell_orders['order_type'].append(order['order_type'])
        else:
            raise ValueError('unsupported side')

    def mod_order(self, order): #M
        # find order
        idx_id, side = self._find_order(order['order_date'], order['order_id'])
        if side == 'buy':
            self.buy_orders['order_date'][idx_id] = order['order_date'] if order['order_date'] != -1 else self.buy_orders['order_date'][idx_id]
            self.buy_orders['priority_date'][idx_id] = order['priority_date'] if order['priority_date'] != -1 else self.buy_orders['priority_date'][idx_id]
            self.buy_orders['order_id'][idx_id] = order['order_id'] if order['order_id'] != -1 else self.buy_orders['order_id'][idx_id]
            self.buy_orders['price'][idx_id] = order['price'] if order['price'] != -1 else self.buy_orders['price'][idx_id]
            self.buy_orders['volume'][idx_id] = order['volume'] if order['volume'] != -1  else self.buy_orders['volume'][idx_id]
            self.buy_orders['order_type'][idx_id] = order['order_type'] if order['order_type'] != -1 else self.buy_orders['order_type'][idx_id]
        if side == 'sell':
            self.sell_orders['order_date'][idx_id] = order['order_date'] if order['order_date'] != -1 else self.sell_orders['order_date'][idx_id]
            self.sell_orders['priority_date'][idx_id] = order['priority_date'] if order['priority_date'] != -1 else self.sell_orders['priority_date'][idx_id]
            self.sell_orders['order_id'][idx_id] = order['order_id'] if order['order_id'] != -1 else self.sell_orders['order_id'][idx_id]
            self.sell_orders['price'][idx_id] = order['price'] if order['price'] != -1 else self.sell_orders['price'][idx_id]
            self.sell_orders['volume'][idx_id] = order['volume'] if order['volume'] != -1 else self.sell_orders['volume'][idx_id]
            self.sell_orders['order_type'][idx_id] = order['order_type'] if order['order_type'] != -1 else self.sell_orders['order_type'][idx_id]

    def del_order(self, order): #D
        # find order
        idx_id, side = self._find_order(order['order_date'], order['order_id'])
        if side == 'buy':
            self.buy_orders['order_date'].pop(idx_id)
            self.buy_orders['priority_date'].pop(idx_id)
            self.buy_orders['order_id'].pop(idx_id)
            self.buy_orders['price'].pop(idx_id)
            self.buy_orders['volume'].pop(idx_id)
            self.buy_orders['order_type'].pop(idx_id)
        if side == 'sell':
            self.sell_orders['order_date'].pop(idx_id)
            self.sell_orders['priority_date'].pop(idx_id)
            self.sell_orders['order_id'].pop(idx_id)
            self.sell_orders['price'].pop(idx_id)
            self.sell_orders['volume'].pop(idx_id)
            self.sell_orders['order_type'].pop(idx_id)

    def clear_orderbook(self): #F
        for k in self.buy_orders:
            self.buy_orders[k].clear()
        for k in self.sell_orders:
            self.sell_orders[k].clear()

    def retransmit_order(self, order): #Y
        try:
            self.mod_order(order)
        except ValueError:
            self.add_order(order)

    def get_buys_df(self, cum=False):
        buys = pd.DataFrame(self.buy_orders).sort_values(['price', 'priority_date'], ascending=[False, True])
        if cum:
            buys = (buys[['price', 'order_type', 'volume']].groupby(['price', 'order_type']).agg(
                {'volume': [np.sum, np.size]})).sort_index(ascending=False)
        return buys

    def get_sells_df(self, cum=False):
        sells = pd.DataFrame(self.sell_orders).sort_values(['price', 'priority_date'])
        if cum:
            sells = (sells[['price', 'order_type', 'volume']].groupby(['price', 'order_type']).agg(
                {'volume': [np.sum, np.size]}))
        return sells

    def get_best_buys(self, num=1, cum=True):
        buys = pd.DataFrame(self.buy_orders).sort_values(['price', 'priority_date'], ascending=[False, True])
        if cum:
            buys_acc = (buys[['price', 'order_type', 'volume']].groupby(['price', 'order_type']).agg(
                {'volume': [np.sum, np.size]})).sort_index(ascending=False)
            buys_p = buys_acc.index.get_level_values('price').values
            buys_v = buys_acc['volume']['sum'].values
            return {'best_buy_prices': buys_p[0:num], 'best_buy_volumes': buys_v[0:num]}
        else:
            return {'best_buy_prices': buys['price'].iloc[0:num].values,
                    'best_buy_volumes': buys['volume'].iloc[0:num].values}

    def get_best_sells(self, num=1, cum=True):
        sells = pd.DataFrame(self.sell_orders).sort_values(['price', 'priority_date'])
        if cum:
            sells_acc = (sells[['price', 'order_type', 'volume']].groupby(['price', 'order_type']).agg(
                {'volume': [np.sum, np.size]}))
            sells_p = sells_acc.index.get_level_values('price').values
            sells_v = sells_acc['volume']['sum'].values
            return {'best_sell_prices': sells_p[0:num], 'best_sell_volumes': sells_v[0:num]}
        else:
            return {'best_sell_prices': sells['price'].iloc[0:num].values,
                    'best_sell_volumes': sells['volume'].iloc[0:num].values}

    def get_bests(self, num=1, cum=True):
        bb = self.get_best_buys(num=num, cum=cum)
        bs = self.get_best_sells(num=num, cum=cum)
        bb.update(bs)
        return bb

    def get_wap(self, level=1, mode='separate'):
        bests = self.get_bests(num=level, cum=True)
        if mode == 'separate':
            assert (len(bests['best_buy_prices']) == len(bests['best_sell_prices']) == level)
            return (bests['best_buy_prices'][level-1] * bests['best_sell_volumes'][level-1] +
                    bests['best_sell_prices'][level-1] * bests['best_buy_volumes'][level-1]) / \
                (bests['best_buy_volumes'][level-1] + bests['best_sell_volumes'][level-1])
        elif mode == 'combine':
            return ((bests['best_buy_prices'] * bests['best_sell_volumes'] +
                     bests['best_sell_prices'] * bests['best_buy_volumes']).sum()) / \
                (bests['best_buy_volumes'] + bests['best_sell_volumes']).sum()
        else:
            raise ValueError('unsupported mode')

    def get_mid(self):
        bests = self.get_bests()
        bpask = bests['best_sell_prices'][0]
        bpbid = bests['best_buy_prices'][0]
        mid = 0.5 * (bpask + bpbid)
        return mid

    def get_spread(self):
        bests = self.get_bests()
        bpask = bests['best_sell_prices'][0]
        bpbid = bests['best_buy_prices'][0]
        spread = bpask - bpbid
        return spread

    def remove_orders_by_volume(self, volume, side):
        ob = self.copy()
        if side == 'buy':
            orders = pd.DataFrame(ob.buy_orders).sort_values(['price', 'priority_date'], ascending=[False, True])
        elif side == 'sell':
            orders = pd.DataFrame(ob.sell_orders).sort_values(['price', 'priority_date'])
        else:
            raise ValueError('unsupported side')

        orders['cum_col'] = orders['volume'].cumsum()
        temp = orders[orders['cum_col'] < volume]
        if len(temp) > 0:
            tvol = temp['cum_col'].iloc[-1]
        else:
            tvol = 0
        for i, row in temp.iterrows():
            ob.del_order(row)
        if tvol < volume and len(temp) < len(orders):
            orders.iat[len(temp), orders.columns.get_loc('volume')] -= (volume - tvol)
            ob.mod_order(orders.iloc[len(temp)])
        return ob

    def remove_orders_by_price(self, price, side):
        ob = self.copy()
        if side == 'buy':
            orders = pd.DataFrame(ob.buy_orders).sort_values(['price', 'priority_date'], ascending=[False, True])
            temp = orders[orders['price'] < price]
        elif side == 'sell':
            orders = pd.DataFrame(ob.sell_orders).sort_values(['price', 'priority_date'])
            temp = orders[orders['price'] > price]
        else:
            raise ValueError('unsupported side')

        #orders['cum_col'] = orders['volume'].cumsum()
        #temp = orders[orders['cum_col'] < volume]
        #if len(temp) > 0:
        #    tvol = temp['cum_col'].iloc[-1]
        #else:
        #    tvol = 0
        for i, row in temp.iterrows():
            ob.del_order(row)
        #if tvol < volume and len(temp) < len(orders):
        #    orders.iat[len(temp), orders.columns.get_loc('volume')] -= (volume - tvol)
        #    ob.mod_order(orders.iloc[len(temp)])
        return ob

    def avg_trade_pirce(self, volume, side):
        if side == 'buy':
            orders = pd.DataFrame(self.buy_orders).sort_values(['price', 'priority_date'], ascending=[False, True])
        elif side == 'sell':
            orders = pd.DataFrame(self.sell_orders).sort_values(['price', 'priority_date'])
        else:
            raise ValueError('unsupported side')
        orders['cum_col'] = orders['volume'].cumsum()
        if orders['cum_col'].iloc[-1] < volume:
            return 0.0
        temp = orders[orders['cum_col'] <= volume]
        prices = temp['price'].values
        volumes = temp['volume'].values
        if len(temp) > 0:
            tvol = temp['cum_col'].iloc[-1]
        else:
            tvol = 0
        if tvol < volume and len(temp) < len(orders):
            prices = np.append(prices, orders['price'].values[len(temp)])
            volumes = np.append(volumes, volume-tvol)
        return np.average(prices, weights=volumes)

    def _find_order(self, order_date, order_id):
        order_pos = None
        # find order in buys
        poss = [p for p, (d, i) in enumerate(zip(self.buy_orders['order_date'], self.buy_orders['order_id'])) if (d == order_date and i == order_id) ]
        if poss:
            order_pos = poss[0]
            if len(poss) > 1:
                raise ValueError('duplicate order buy')
            return order_pos, 'buy'
        else:
            # find order in buys
            poss = [p for p, (d, i) in enumerate(zip(self.sell_orders['order_date'], self.sell_orders['order_id'])) if (d == order_date and i == order_id)]
            if poss:
                order_pos = poss[0]
                if len(poss) > 1:
                    raise ValueError('duplicate order sell')
                return order_pos, 'sell'
            else:
                if self.verbose:
                    print(order_date, order_id)
                raise ValueError('order not found')
