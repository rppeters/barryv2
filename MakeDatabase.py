from Database import Database
from Coindata import Coindata
from Result import Result

class makeDatabase:
    def __init__(self):
        self.db = Database()
        self.coins = []
        self.time_frames = ['1h','2h','4h','6h','8h','12h','1d']

    def initializeDB(self):
        for tf in self.time_frames:
            for coin in self.coins:
                cd = Coindata(coin, tf)
                self.db.addData(cd)
                rs = cd.findDivergences()
                for r in rs:
                    self.db.addResult(r)


    