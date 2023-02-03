import torch

def Example():
    x = torch.randn(2, 2, requires_grad=True)
    
    # Scalar outputs
    out = x.sum()  # Size([])
    batched_grad = torch.arange(3)  # Size([3])
    grad, = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
    
    # loop approach
    grads = torch.stack(([torch.autograd.grad(out, x, torch.tensor(a))[0] for a in range(3)]))

def f(x):
    return torch.square(x)

def df(x):
    return 2 * x

def ComputeBatchedGradients1(print=False):
    global x, out_sq, N, C
    grad = torch.autograd.grad(out_sq, x, grad_outputs=torch.ones_like(out_sq), create_graph=True)
    if print:
        print(grad)
        print(df(x))

def ComputeBatchedGradients2(print=False):
    global x, out_sq, N, C
    out_sq = out_sq.sum()  # Scalarize out
    grad = torch.autograd.grad(out_sq, x, grad_outputs=torch.tensor([1]), create_graph=True, is_grads_batched=True)
    if print:
        print(grad)
        print(df(x))

def ComputationTime():
    import timeit
    print(timeit.timeit("ComputeBatchedGradients1()", setup="from __main__ import ComputeBatchedGradients1", number=1000))
    print(timeit.timeit("ComputeBatchedGradients2()", setup="from __main__ import ComputeBatchedGradients2", number=1000))

def g(x):
    return x @ weights

if __name__ == '__main__':
    device = 'cuda'
    N = 1024
    C = 1
    torch.random.manual_seed(0)
    x = torch.rand((N, C), requires_grad=True, device=device)
    weights = torch.arange(C * 3, device=device, dtype=x.dtype).reshape(C, -1)  # 2D out
    
    out_sq = f(x) # Square
    out_lin = g(x)
    
    # grad_out=torch.zeros_like(out_lin)
    # grad_out[:,0]=1 # output dimension의 첫번째
    #
    # grad_out = torch.zeros_like(out_lin)
    # grad_out[:, 1] = 2  # d out_lin[1]/d x 만을 계산
    
    grad_out=torch.ones_like(out_lin[:,2])
    grad_lin, = torch.autograd.grad(out_lin[:,2], x, grad_outputs=grad_out)
    """
    torch.autograd.grad
    If multidimensional input, output given, returns sum of output derivatives(=num of outputs) -> returns gradient with input shape
    다차원텐서 미분 시 차원간 합이 반환되므로, 그냥 벡터 연산만 사용하는게 가장 간결하고 명확할 듯.
    """
    print(out_lin)
    print(out_lin.shape)
    print()
    print(grad_lin)
    print(grad_lin.shape)
    print()
    print(weights)
    print(weights.shape)
