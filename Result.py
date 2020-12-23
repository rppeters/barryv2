
class Result:
    def __init__(self, coin, type, time_frame, score, beg, end):
        self.coin = coin
        self.type = type
        self.time_frame = time_frame
        self.score = score
        self.beg = beg #candles away from present
        self.end = end #cadles away from present

    def __str__(self):
        return "Pairing: {} | Time Frame: {} | Type Divergence: {} | Score: {} | When: {} to {} periods ago".format(self.coin, self.time_frame, self.type, self.score, self.beg, self.end)
