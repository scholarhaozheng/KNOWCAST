from metro_components.Line import Line


class LineManager:
    lines = {}
    adj_mx = []
    station_sequence = []

    def add_line(self, line_number, new_station):
        if line_number in self.lines:
            self.lines[line_number].add_station(new_station)
        else:
            line = Line(line_number)
            line.add_station(new_station)
            self.lines[line_number] = line

    def print_line_info(self, line_number):
        print("Line: ", line_number)
        print("lines: ", end="")
        if line_number in self.lines:
            line = self.lines[line_number]
            for each_station in line.stations:
                print(each_station, end="->")
        print()

    def print_all_info(self):
        for each in self.lines:
            self.print_line_info(self.lines[each].line_number)
            print()
        print("Line count ", len(self.lines))

    def print_stops(self, line_number, from_stop, to_stop):
        line = self.lines[line_number]
        start_index = 0
        end_index = 0
        stations = line.stations
        for i in range(0, len(stations)):
            if stations[i].name == from_stop:
                start_index = i
            elif stations[i].name == to_stop:
                end_index = i

        if start_index > end_index:
            start_printing = False
            for each in reversed(stations):
                if each.name == from_stop:
                    print(each.name, " -> ", end="")
                    start_printing = True
                elif each.name == to_stop:
                    print(each.name)
                    start_printing = False
                elif start_printing:
                    print(each.name, " -> ", end="")
        else:
            start_printing = False
            for each in stations:
                if each.name == from_stop:
                    print(each.name, " -> ", end="")
                    start_printing = True
                elif each.name == to_stop:
                    print(each.name)
                    start_printing = False
                elif start_printing:
                    print(each.name, " -> ", end="")

        print()


