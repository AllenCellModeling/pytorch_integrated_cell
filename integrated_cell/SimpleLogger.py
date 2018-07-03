class SimpleLogger:
    def __init__(self, fields, print_format=''):
        if isinstance(print_format, str) and not print_format:
            printstr = ''
            for field in fields:
                printstr = printstr + field + ': %f '
            print_format = printstr
            
        self.print_format = print_format

        self.fields = fields
        
        self.log = dict()
        for field in fields:
            self.log[field] = []
            
    def add(self, input):
        assert(len(input) == len(self.fields))
        
        for i in range(0, len(self.fields)):
            self.log[self.fields[i]].append(input[i])
            
        if isinstance(self.print_format, str):
            print(self.print_format % input)