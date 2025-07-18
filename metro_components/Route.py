

class Route:

    def __init__(self):
        self.from_stop = ""
        self.to_stop = ""
        self.line_number = 0
        self.stops = 9999
        self.stations = []
        self.start_index = 0
        self.end_index = 0
        self._track_changes = False

    def __setattr__(self, name, value):
        if hasattr(self, '_track_changes') and self._track_changes and name != '_track_changes':
            print(f"Route attribute changed: {name} = {value}")
        super().__setattr__(name, value)

    def construct_route(self, line_manager, station_manager, from_station, to_station, lines):
        self.from_stop = from_station.name
        self.to_stop = to_station.name
        self.start_index = from_station.index
        self.end_index = to_station.index
        self.stops = 9999
        if len(lines) == 0:
            self.stops = 9999
            return self
        else:
            for each_line in lines:
                line = line_manager.lines[each_line]
                start_index = 0
                stop_index = 0
                find_start_index = False
                find_stop_index = False

                for i in range(0, len(line.stations)):
                    if line.stations[i].name == from_station.name:
                        start_index = i
                        find_start_index = True
                    if line.stations[i].name == to_station.name:
                        stop_index = i
                        find_stop_index = True
                    if find_start_index and find_stop_index:
                        break

                stops = abs(start_index - stop_index)

                if stops < self.stops:
                    self.stops = stops
                    self.line_number = line.line_number
                    if start_index <= stop_index:
                        station_list = line.stations[start_index:stop_index + 1]
                    else:
                        station_list = line.stations[stop_index:start_index + 1][::-1]
                    for station in station_list:
                        self.stations.append(station_manager.stations[station.name])

