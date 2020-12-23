import numpy
import asyncio
import aiohttp as aiohttp
import matplotlib 
from matplotlib import pyplot as plt
from Result import Result


class Coindata:
    LIMIT = 42 #number of candles for analysis
    def __init__(self, coin, time_frame):
        self.coin = coin
        self.time_frame = time_frame
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.getData()) #list of 120 dictionaries/candles

    async def getData(self):
        ENDPOINT = 'https://api.binance.com/api/v1/klines'
        url = ENDPOINT + '?symbol=' + self.coin + '&interval=' + self.time_frame + '&limit=' + str(120)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    coin_data = await resp.json()

        self.data = [{
            "time": int(d[0]),
            "open": float(d[1]),
            "close": float(d[4]),
            "volume": float(d[5])
        } for d in coin_data]

    def addNewest(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.addNewest_helper())

    async def addNewest_helper(self):
        ENDPOINT = 'https://api.binance.com/api/v1/klines'
        url = ENDPOINT + '?symbol=' + self.coin + '&interval=' + self.time_frame + '&limit=1'
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    coin_data = await resp.json()

        coin_data = coin_data[0]
        print(coin_data)
        if coin_data[0] != self.data[-1]["time"]: #remove oldest, add newest
            self.data = self.data[1:120].append(coin_data)
        else: #otherwise, update last
            self.data[-1] = {
                "time": int(coin_data[0]),
                "open": float(coin_data[1]),
                "close": float(coin_data[4]),
                "volume": float(coin_data[5])
            }

    def calculateRSI(self):
        #Download and organize Price data into list
        changes = [d['close'] - d['open'] for d in self.data]

        #create total avg gain and loss lists with correct 0 values based on gains and losses
        gains = []
        losses = []
        for c in changes:
            if c >= 0:
                gains.append(abs(c))
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(c))

        #calculate RS based off change and initialize last_avg_gain and last_avg_loss 
        list_rs = []
        avg_gain = numpy.mean(gains[0:14])
        avg_loss = numpy.mean(losses[0:14])
        list_rs.append(avg_gain / avg_loss)
        prev_AG = avg_gain
        prev_AL = avg_loss
        data_len = len(self.data)
        for i in range(14,data_len):
            avg_gain = ((prev_AG * 13) + gains[i]) / 14
            avg_loss = ((prev_AL * 13) + losses[i]) / 14
            list_rs.append(avg_gain/avg_loss)
            prev_AG = avg_gain
            prev_AL = avg_loss
            #for void price calculations; still confused on use
            if i == data_len-2:            
                last_avg_gain = prev_AG
                last_avg_loss = prev_AL

        #calculate RSI 
        list_RSI=[]
        for rs in list_rs:
            list_RSI.append(round((100 - (100 / (1 + rs))),2))

        #reduce list_RSI to last 42 rsi's for analysis
        return list_RSI[-self.LIMIT:],last_avg_gain,last_avg_loss

    def calculateOBV(self):
        #Initialize list_OBV with starting value at 0 and prev with an arbitrary value of 0 (removed later)
        list_OBV = [0]
        prev = 0
        #Calculate OBV
        for i in range(1,len(self.data)):
            change_day = self.data[i]['close'] - self.data[i-1]['close']
            change_volume = int(self.data[i]['volume']) #float or int volume?

            if change_day > 0.0:
                prev = prev + change_volume
            elif change_day < 0:
                prev = prev - change_volume
            list_OBV.append(prev)
        
        del list_OBV[0] #Remove the 0 from the beginning of the list
        return list_OBV[-self.LIMIT:]

    def calculateMACD(self):
        #Retrieve closes from data
        prices = [d['close'] for d in self.data]
        #Set up constants for a 12 EMA and calculate
        scontant_12 = 2 / 13
        ema_12 = []
        #initial calculation
        prev_ema = numpy.mean(prices[0:12])
        ema_12.append(prev_ema)
        
        for p in prices[12:]:
            prev_ema = (p - prev_ema) * scontant_12 + prev_ema
            ema_12.append(prev_ema)
                
        #Set up constants for a 26 EMA and calculate
        sconstant_26 = 2 / 27
        ema_26 = []
        prev_ema = numpy.mean(prices[0:26])
        ema_26.append(prev_ema)
        for p in prices[26:]:
            prev_ema = (p - prev_ema) * sconstant_26 + prev_ema
            ema_26.append(prev_ema)
                
        #Calculate MACD Line
        list_macd = [ema_12[-i] - ema_26[-i] for i in range(1,56)][::-1]

        #Calculate signal line
        sconstant_9 = 1 / 10
        list_sigline = []

        prev_ema = numpy.mean(list_macd[0:9])
        list_sigline.append(prev_ema)
        for macd in list_macd[9:55]:
            prev_ema = (macd - prev_ema) * sconstant_9 + prev_ema
            list_sigline.append(prev_ema)

        return list_macd[-self.LIMIT:], list_sigline[-self.LIMIT:]

    def findPrices(self):
        #Returns lowest price values for each candle
        prices = [d['open'] if d['open'] <= d['close'] else d['close'] for d in self.data][-self.LIMIT:]
        threshold = numpy.mean(prices) * 0.0005
        for i in range(1,len(prices)):
            if (abs(prices[i] - prices[i-1]) <= threshold):
                prices[i] = prices[i-1]
        return prices

    def findLows(self):
        prices = self.findPrices()

        #pass1
        lows = []
        lows_idx = []
        for i in range(1,len(prices) - 1):
            if prices[i] <= prices[i-1] and prices[i] <= prices[i+1] and prices[i] != prices[i-1]:
                lows.append(prices[i])
                lows_idx.append(i)
                
        #pass2
        lows2 = [lows[0]]
        lows2_idx = [lows_idx[0]]
        for i in range(1,len(lows)-1):
            if lows[i] < lows[i - 1]:
                lows2.append(lows[i])
                lows2_idx.append(lows_idx[i])
        if lows[-1] < lows[-2]:
            lows2.append(lows[-1])
            lows2_idx.append(lows_idx[-1])

        #pass3
        lows3 = [lows2[0]]
        lows3_idx = [lows2_idx[0]]
        for i in range(1, len(lows2)):
            if (lows2[i] < lows2[i - 1]): #if low decreases from previous low
                lows3.append(lows2[i])
                lows3_idx.append(lows2_idx[i])

        return lows2, lows2_idx

    def findDivergences(self):
        #Returns list of Result classes
        lows, lows_idx = self.findLows()
        ll_RSI, ll_OBV, ll_macd, ll_sigline = ([],[],[],[])
        list_RSI,lag,lal = self.calculateRSI()
        list_OBV = self.calculateOBV()
        list_macd, list_sigline = self.calculateMACD()
        #Add correct RSIs and OBVs according to ll_idx
        for i in lows_idx:
            ll_RSI.append(list_RSI[i])
            ll_OBV.append(list_OBV[i])
            ll_macd.append(list_macd[i])
            ll_sigline.append(list_sigline[i])

        score_threshold = 1 #filter for low scores/miniscule divergences
        results = []
        for i in range(len(lows) - 1):
            if lows[i] > lows[i+1]: #if lows decrease then trend is moving down

                score_price = (((lows[i] - lows[i+1])  /  lows[i]) * 100) 
                #check for RSI Divergence
                if ll_RSI[i] < ll_RSI[i+1]:
                    score = round((ll_RSI[i+1] - ll_RSI[i]) * score_price,2) #RSI change * percent price change
                    if score >= score_threshold:
                        results.append(Result(self.coin, "RSI", self.time_frame, score, 42 - lows_idx[i], 42 - lows_idx[i+1]))
                #check for OBV Divergence
                if ll_OBV[i] < ll_OBV[i]:
                    score = round((((ll_OBV[i+1] - ll_OBV[i]) / numpy.average(list_OBV[-self.LIMIT:])) * 100) * score_price,2) #volume increase compared to relative average * percent price change
                    if score >= score_threshold:
                        results.append(Result(self.coin, "OBV", self.time_frame, score, 42 - lows_idx[i], 42 - lows_idx[i+1])) 
                #check for MACD Divergence
                if ll_macd[i] < ll_macd[i+1] and ll_sigline[i] < ll_sigline[i+1]:
                    score = round(((ll_sigline[i+1] - ll_sigline[i]) / ll_sigline[i]) * score_price  * 10,2) #signal line percent increase * percent price change * weight
                    if score >= score_threshold:
                        results.append(Result(self.coin, "MACD", self.time_frame, score, 42 - lows_idx[i], 42 - lows_idx[i+1]))
        return results 

    def printData(self):
        for d in self.data:
            print(d)

if __name__ == "__main__":
    coin = "VIBEBTC"
    tf = "1h"


    cd = Coindata(coin, tf)
    cd.printData()
    cd.addNewest()
    cd.printData()

    test =  [Result("A", "MACD", "tf", 5.5,1,2),
    Result("A", "MACD", "tf", 1.25,1,2),
    Result("A", "MACD", "tf", 8.2,1,2),
    Result("A", "MACD", "tf", 82.5,1,2),
    ]
    tested = sorted(test, key=lambda test: test.score, reverse=True)
    for t in tested:
        print(t)


    '''
    rsi,lag,lal = cd.calculateRSI()
    obv = cd.calculateOBV()
    macd, sigline = cd.calculateMACD()
    results = cd.findDivergences()
    lows, lows_idx = cd.findLows()
    prices = cd.findPrices()

    coors = [(lows_idx[i],lows[i]) for i in range(len(lows))]

    counter =0
    for p in prices:
        print(counter, " " , p)
        counter += 1

    fig, ax = plt.subplots()
    ax.plot(range(len(prices)), prices, color='red',label='Prices')
    for coor in coors:
        ax.add_artist(plt.Circle(coor, 0.36, color='blue'))
    ax.legend()
    ax.grid()
    plt.show()'''

