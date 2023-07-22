import torch
import poptorch


class Calculator(torch.nn.Module):
    def __init__(self):
        super(Calculator, self).__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, val):
        return self.lin(val)


class State(torch.nn.Module):
    def __init__(self):
        super(State, self).__init__()
        self.register_buffer("accu", torch.tensor([4.], dtype=torch.float))

    def update(self, val):
        self.accu = val
        #self.accu.copy_(val)


class Simulator(torch.nn.Module):
    def __init__(self, state, calculator):
        super(Simulator, self).__init__()
        self.register_module("state", state)
        self.register_module("calc", calculator)

    def loop(self, val):
        val = val + self.state.accu
        val = self.calc(val)
        self.state.accu.copy_(val)
        #self.state.update(val)
        return [val]

    def forward(self, steps):
        val = self.state.accu
        #for _ in range(steps):
        #    val = self.loop([val])[0]
        #return val
        return poptorch.for_loop(steps, self.loop, [val])


calc = Calculator()
state = State()
sim = Simulator(state, calc)
print(sim(10))
sim = poptorch.inferenceModel(sim)
print(sim(10))
