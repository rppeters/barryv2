class Database:
    def __init__(self):
        self.data = {
            '1h':[],
            '2h':[],
            '4h':[],
            '6h':[],
            '8h':[],
            '12h':[],
            '1d':[],
        }
        self.results = {
            '1h':[],
            '2h':[],
            '4h':[],
            '6h':[],
            '8h':[],
            '12h':[],
            '1d':[],
        }

    def addResult(self, r):
        self.results[r.time_frame].append(r)
    
    def addData(self, d):
        self.data[d.time_frame].append(d)
    
    def sort(self, b_reverse):
        for tf,rs in self.results:
            self.results[tf] = sorted(rs, key=lambda rs: rs.score, reverse=b_reverse)

    def recent(self, tf):
        return [r for r in self.results[tf] if r.end == 1]

    def count(self, tf): 
        return len(self.results[tf])

    def triple_divergences(self, tf):
        tdivs = []
        