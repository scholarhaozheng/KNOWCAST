class Line:

    line_number = 0
    stations = []

    def __init__(self, line_number):
        self.line_number = line_number
        self.stations = []

    def add_station(self, station):
        self.stations.append(station)

