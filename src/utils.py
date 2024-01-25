import torch

def get_parameterizable_modules(model):
    parameterizable_modules=[]
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            parameterizable_modules.append(module)
        elif isinstance(module, torch.nn.Conv2d):
            parameterizable_modules.append(module)

    return parameterizable_modules

def init_SP(model):
    parameterizable_modules =get_parameterizable_modules(model)

    for module in parameterizable_modules:
        # fan_in and fan_out in μP and PyTorch nomenclature are different
        # The bias and the weights have different fan_in (,fan_out)

        fan_in_bias = 1
        fan_out_bias = module.weight.shape[0]
        torch.nn.init.zeros_(module.bias)

        fan_in_weight = module.weight.shape[1]
        fan_out_weight = module.weight.shape[0]
        torch.nn.init.normal_(module.weight, mean=0, std=1/fan_in_weight**(1/2))

def init_μP(proxy, target):
    proxy_modules = get_parameterizable_modules(proxy)
    target_modules = get_parameterizable_modules(target)

    # Input+hidden
    for i in range(len(target_modules)-1):
        # fan_in and fan_out in μP and PyTorch nomenclature are different
        # The bias and the weights have different fan_in (,fan_out)

        fan_in_bias0 = 1
        fan_out_bias0 = proxy_modules[i].bias.shape[0]
        fan_in_bias = 1
        fan_out_bias = target_modules[i].bias.shape[0]
        torch.nn.init.zeros_(target_modules[i].bias)

        fan_in_weight0 = proxy_modules[i].weight.shape[1]
        fan_out_weight0 = proxy_modules[i].weight.shape[0]
        fan_in_weight = target_modules[i].weight.shape[1]
        fan_out_weight = target_modules[i].weight.shape[0]
        torch.nn.init.normal_(target_modules[i].weight, mean=0, std=1/fan_in_weight**(1/2))

    # Output
    fan_in_bias0 = 1
    fan_out_bias0 = proxy_modules[-1].bias.shape[0]
    fan_in_bias = 1
    fan_out_bias = target_modules[-1].bias.shape[0]
    torch.nn.init.zeros_(target_modules[-1].bias)

    fan_in_weight0 = proxy_modules[-1].weight.shape[1]
    fan_out_weight0 = proxy_modules[-1].weight.shape[0]
    fan_in_weight = target_modules[-1].weight.shape[1]
    fan_out_weight = target_modules[-1].weight.shape[0]
    torch.nn.init.normal_(target_modules[-1].weight, mean=0, std=fan_in_weight0**(1/2)/fan_in_weight)

class Adam_μP(torch.optim.Adam):
	def __init__(self, proxy, target, lr):
		proxy_modules = get_parameterizable_modules(proxy)
		target_modules = get_parameterizable_modules(target)

		params = []

		# Input
		# fan_in and fan_out in μP and PyTorch nomenclature are different
        # The bias and the weights have different fan_in (,fan_out)

		fan_in_bias0 = 1
		fan_out_bias0 = proxy_modules[0].bias.shape[0]
		fan_in_bias = 1
		fan_out_bias = target_modules[0].bias.shape[0]
		params.append({"params": target_modules[0].bias, "lr": lr})

		fan_in_weight0 = proxy_modules[0].weight.shape[1]
		fan_out_weight0 = proxy_modules[0].weight.shape[0]
		fan_in_weight = target_modules[0].weight.shape[1]
		fan_out_weight = target_modules[0].weight.shape[0]
		params.append({"params": target_modules[0].weight, "lr": lr})

		# Hidden+output
		for i in range(1, len(target_modules)):
			fan_in_bias0 = 1
			fan_out_bias0 = proxy_modules[i].bias.shape[0]
			fan_in_bias = 1
			fan_out_bias = target_modules[i].bias.shape[0]
			params.append({"params": target_modules[i].bias, "lr": lr})

			fan_in_weight0 = proxy_modules[i].weight.shape[1]
			fan_out_weight0 = proxy_modules[i].weight.shape[0]
			fan_in_weight = target_modules[i].weight.shape[1]
			fan_out_weight = target_modules[i].weight.shape[0]
			params.append({"params": target_modules[i].weight, "lr": lr*fan_in_weight0/fan_in_weight})

		super().__init__(params)