class Project:
    def __init__(self, symbol: str, name: str = None, floor_price: float = None, 
                 volume: float = None, has_cnfts: bool = False):
        self.symbol = symbol
        self.name = name
        self.floor_price = floor_price
        self.volume = volume
        self.has_cnfts = has_cnfts 