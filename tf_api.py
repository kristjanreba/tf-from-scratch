import numpy

class Graph():
    def __init__(self):
        self.placeholders = []
        self.variables = []
        self.operations = []
        self.constants = []

    def as_default(self):
        global _default_graph
        _default_graph = self


class Operation():
    def __init__(self, input_nodes=None):
        self.input_nodes = input_nodes
        self.output = None
        _default_graph.operations.append(self)

    def farward(self):
        pass

    def backward(self):
        pass


class BinaryOperation(Operation):
    def __init__(self, a, b):
        super().__init__([a,b])


class add(BinaryOperation):
    def forward(self, a, b):
        return a + b

    def backward(self, upstream_grad):
        raise NotImplementedError


class multiply(BinaryOperation):
    def forward(self, a, b):
        return a * b

    def backward(self, upstream_grad):
        raise NotImplementedError


class divide(BinaryOperation):
    def forward(self, a, b):
        return a / b

    def backward(self, upstream_grad):
        raise NotImplementedError


class matmul(BinaryOperation):
    def forward(self, a, b):
        return a.dot(b)

    def backward(self, upstream_grad):
        raise NotImplementedError


class Placeholder(object):
    def __init__(self):
        self.value = None
        _default_graph.placeholders.append(self)


class Constant(object):
    def __init__(self, value=None):
        self.__value = value
        _default_graph.constants.append(self)

        @property
        def value(self):
            return self.__value

        @value.setter
        def value(self, value):
            raise ValueError("Cannot reassign value.")


class Variable(object):
    def __init__(self, initial_value):
        self.initial_value = initial_value
        _default_graph.placeholders.append(self)


def topology_sort(operation):
    ordering = []
    visited_nodes = set()

    def recursive_helper(node):
      if isinstance(node, Operation):
        for input_node in node.input_nodes:
          if input_node not in visited_nodes:
            recursive_helper(input_node)

      visited_nodes.add(node)
      ordering.append(node)

    # start recursive depth-first search
    recursive_helper(operation)

    return ordering


class Session():
    def run(self, operation, feed_dict={}):
        nodes_sorted = topology_sort(operation)

        for node in nodes_sorted:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable or type(node) == Constant:
                node.output = node.value
            else:
                inputs = [node.output for node in node.input_nodes]
                node.output = node.node.forward(*inputs)

        return operation.output
