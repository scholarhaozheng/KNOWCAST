import copy

class Path:

    def __init__(self):
        self.from_stop = ""
        self.to_stop = ""
        self.start_index=0
        self.end_index=0
        self.routes = []
        self.station_visit_sequence_index=[]
        self.station_visit_sequence = []
        self.path_number = 0

    def __lt__(self, other):
        # This method always returns False, regardless of the circumstances.
        # This means that when distance_plus_bias is equal, the insertion order will be preserved.
        return False

    def add_path(self, route):
        new_path=copy.deepcopy(self)
        new_path.to_stop=route.to_stop
        new_path.end_index=route.end_index
        if (len(new_path.routes)==0):
            new_path.from_stop = route.from_stop
            new_path.start_index = route.start_index
            new_path.routes.append(route)
            new_path.station_visit_sequence = new_path.station_visit_sequence + route.stations
            new_path.station_visit_sequence_index = new_path.station_visit_sequence_index + [int(station.index) for
                                                                                             station in
                                                                                             route.stations]

        else:
            if new_path.station_visit_sequence_index[-1] != route.start_index:
                new_path.station_visit_sequence = new_path.station_visit_sequence + route.stations
                new_path.station_visit_sequence_index = new_path.station_visit_sequence_index + [int(station.index)
                                                                                                 for
                                                                                                 station in
                                                                                                 route.stations]
            else:
                new_path.station_visit_sequence = new_path.station_visit_sequence[:-1] + route.stations
                new_path.station_visit_sequence_index = new_path.station_visit_sequence_index[:-1] + [
                    int(station.index)
                    for
                    station in
                    route.stations]
            if (new_path.routes[-1].line_number == route.line_number):
                route_add_fregment = copy.deepcopy(new_path.routes[-1])
                route_add_fregment.to_stop = route.to_stop
                route_add_fregment.end_index = route.end_index
                route_add_fregment.stops += route.stops
                route_add_fregment.stations.extend(route.stations[1:])
                del new_path.routes[-1]
                new_path.routes.append(route_add_fregment)
            else:
                new_path.routes.append(route)
        return new_path




