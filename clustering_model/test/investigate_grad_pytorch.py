import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter


class Classfication(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Classfication, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('bias', None)
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 0)

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.softmax(
            torch.matmul(input, self.weight), dim=1
        )

        # return torch.matmul(input, self.weight)


model = Classfication(8, 2)
inputs = torch.FloatTensor([[0, 0, 1, 0, 0, 0, 0, 0]])
y_pred = model(inputs)
y_true = torch.FloatTensor([[0, 1]])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss = nn.BCELoss()(y_pred, y_true)
# loss = nn.BCEWithLogitsLoss()(y_pred, y_true)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'y_pred: {y_pred}, y_true: {y_true}')
print(f'loss: {loss}')
for param in model.parameters():
    print(param.grad)