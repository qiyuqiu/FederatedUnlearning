class Rank:
    def __init__(self, rank_id, proj):
        self.proj = proj
        self.clients = []
        self.rank_id = rank_id
        self.model_before_train = None
        self.clients_before = []
