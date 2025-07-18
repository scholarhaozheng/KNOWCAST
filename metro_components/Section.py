class Section():
    def __init__(self, start_station, terminal_station, section_line):
        self.start_station = start_station
        self.terminal_station = terminal_station
        self.section_line = section_line
        self.depart_freq = 0
        self.visit_prob = 0

    def __hash__(self):
        return hash(self.start_station.name + self.terminal_station.name)

    def __eq__(self, other):
        return (self.start_station.name, self.terminal_station.name) == (other.start_station.name, other.terminal_station.name)