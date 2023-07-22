import torch
import poptorch

example_input = torch.tensor([
    [0.3283, -0.2885,  0.1700],
    [0.0632, -0.0703,  0.5984],
    [0.3099,  0.3316,  0.5361],
    [0.8396,  0.0650,  0.2301],
    [0.6964, -0.0667,  0.7110],
    [0.2467, -0.4019,  0.0623],
    [0.5078, -0.0873,  0.1596],
    [0.3865, -0.3983,  0.1749]
])


class NormModel(torch.nn.Module):
    def forward(self, inputs):
        inputs.requires_grad_()
        norm_result = inputs.norm(dim=1)
        grad = torch.autograd.grad(norm_result.sum(), inputs)
        return norm_result, grad


class LinalgNormModel(torch.nn.Module):
    def forward(self, inputs):
        inputs.requires_grad_()
        norm_result = inputs.norm(dim=1)
        grad = torch.autograd.grad(norm_result.sum(), inputs)
        return norm_result, grad


class CustomModel(torch.nn.Module):
    def forward(self, inputs):
        inputs.requires_grad_()
        norm_result = inputs.pow(2).sum(-1).sqrt()
        grad = torch.autograd.grad(norm_result.sum(), inputs)
        return norm_result, grad


norm_model = NormModel()
linalg_model = LinalgNormModel()
customNorm_model = CustomModel()

norm_model = poptorch.inferenceModel(norm_model)
linalg_model = poptorch.inferenceModel(linalg_model)
customNorm_model = poptorch.inferenceModel(customNorm_model)

try:
    print(norm_model(example_input))
except Exception as exc:
    print(exc)

try:
    print(linalg_model(example_input))
except Exception as exc:
    print(exc)

try:
    print(customNorm_model(example_input))
except Exception as exc:
    print(exc)
