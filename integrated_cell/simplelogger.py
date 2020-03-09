class SimpleLogger:
    def __init__(self, fields, print_format=""):
        if isinstance(print_format, str) and not print_format:
            printstr = ""
            for field in fields:
                printstr = printstr + field + ": %f "
            print_format = printstr

        self.print_format = print_format

        self.fields = fields

        self.log = dict()
        for field in fields:
            self.log[field] = []

    def add(self, inputs):

        assert len(inputs) == len(self.fields)

        for i in range(0, len(self.fields)):
            self.log[self.fields[i]].append(inputs[i])

        if isinstance(self.print_format, str):
            print(self.print_format % tuple(inputs))

    def __len__(self):
        return len(self.log[self.fields[0]])
