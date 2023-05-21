class Organ():
    def __init__(self, organ_type=""):
        self.organ_type = organ_type
        self.vertices = []
        self.junctions = []
        self.edges = []
        # self.faces = []
        # self.mesh = []