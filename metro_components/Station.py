class Station:

    name = ""
    lines = []
    index = 0
    station_sequence_of_the_line = {}

    def __init__(self, name, index, lat_lng):
        self.name = name
        self.lines = []
        self.lat_lng = lat_lng
        self.index = index
        self.station_sequence_of_the_line = {}

    def add_line(self, line_number):
        self.lines.append(line_number)


