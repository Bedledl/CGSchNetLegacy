import logging

import schnetpack
import torch
import poptorch
from schnetpack.model import AtomisticModel

from create_model import create_model
from test.example_schnet_input import example_schnet_input

log = logging.getLogger(__name__)


def deactivate_postprocessing(model: AtomisticModel) -> AtomisticModel:
    if hasattr(model, "postprocessors"):
        for pp in model.postprocessors:
            if isinstance(pp, schnetpack.transform.AddOffsets):
                log.info("Found `AddOffsets` postprocessing module...")
                log.info(
                    "Constant offset of {:20.11f} per atom  will be removed...".format(
                        pp.mean.detach().cpu().numpy()
                    )
                )
    model.do_postprocessing = False
    return model


class Simulator(torch.nn.Module):
    def __init__(self, model):
        super(Simulator, self).__init__()
        self.register_module("model", model)
        #self.model = model


    def loop(self, energies):
        dict_result = self.model(example_schnet_input)
        energies_new = dict_result["energy"]
        return energies + energies_new

    def forward(self, steps):
        r = torch.tensor([0])
        for _ in range(steps):
            r = self.loop(r)
        #r = poptorch.for_loop(steps, self.loop, [torch.tensor([0])])[0]
        return r

    def to(self, arg):
        new_self = super(Simulator, self).to(arg)
        new_self.model = new_self.model.to(arg)
        return new_self


def main():
    #model = torch.load("train_chignolin/best_inference_model", map_location="cpu").to(torch.float32)
    #model.eval()

    model = create_model(
        32,
        20,
        5
    )
    model.eval()
    model = deactivate_postprocessing(model)


    ipu_model = poptorch.inferenceModel(model)

    running_ipu = True

    if running_ipu:
        dict_result = ipu_model(example_schnet_input)

    else:
        with torch.autograd.profiler.profile(with_stack=True, profile_memory=True,
                                         experimental_config=torch._C._profiler._ExperimentalConfig(
                                                verbose=True)) as prof:
            dict_result = model(example_schnet_input)
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))

    print(dict_result)
    return


    s = Simulator(model)
    s.to(torch.float32)
    print(s.get_submodule("model"))
    print(next(model.parameters()).device)
    #s = poptorch.inferenceModel(s)
    #print(next(s.model.parameters()).device)
    #s.model.to(torch.float32)
    #s.model.to(torch.device("ipu:0"))
    #print(
    #    s(3)
    #)


if __name__ == '__main__':
    main()