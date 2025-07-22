from metro_components.Station import Station

class StationManager:

    stations = {}
    station_index = {}
    index_station = {}
    station_names = set()
    station_distance_matrix = []

    def add_station(self, name, line, station_index, lat_lng):
        self.station_names.add(name)
        if name in self.stations:
            self.stations[name].add_line(line)
        else:
            station = Station(name, station_index, lat_lng)
            station.add_line(line)
            self.stations[name] = station
            self.station_index[name] = station_index

    def print_station_info(self, name, index):
        print("Station: ", name, "-> ", end="")
        print("station_index: ", index, "-> ", end="")
        print("lines: ", end="")
        if name in self.stations:
            station = self.stations[name]
            for each_line in station.lines:
                print(each_line, end=", ")

    def print_all_info(self):
        for each in self.stations:
            self.print_station_info(self.stations[each].name,self.stations[each].index)
            print()
        print("Station count ", len(self.stations))

