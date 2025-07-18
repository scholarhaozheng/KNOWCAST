
class OD():
    def __init__(self, start_station_name, terminal_station_name):
        self.start_station_name = start_station_name
        self.terminal_station_name = terminal_station_name
        self.sec_vst_freq_dic = {}
        self.pass_sections_list = []

    def __hash__(self):
        return hash(self.start_station_name + self.terminal_station_name)

    def __eq__(self, other):
        return (self.start_station_name, self.terminal_station_name) == (other.start_station_name, other.terminal_station_name)
