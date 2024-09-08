import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias):
        # weight shape - output x input dimension
        # bias shape - output dimension

        # implement y = x (mult) w_transpose + b
        weight_T = torch.transpose(weight, 0, 1)
        y = torch.matmul(input, weight_T) + bias
        # YOUR IMPLEMENTATION HERE!
        # output=1 is a placeholder
        output = y

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        # save the tensors required for back pass
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Shapes.
        # grad_output - batch x output_count
        # grad_input  - batch x input
        # grad_weight - output x input
        # grad_bias   - output shape

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_output_t = torch.transpose(grad_output, 0, 1)

        #input_t = torch.transpose(input, 0, 1)
        #weight_t = torch.transpose(weight, 0, 1)

        grad_weight = torch.matmul(grad_output_t,input)
        grad_bias = grad_output 
        grad_input = torch.matmul(grad_output,weight)
        # use either print or logger to print its outputs.
        # make sure you disable before submitting
        # print(grad_input)
        # logger.info("grad_output: %s", grad_bias.shape)

        return grad_input, grad_weight, grad_bias
    
class CustomReLULayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # output=1 is a placeholder
        output = torch.where(input < 0.0, 0.0, input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = grad_output.clone()

        # Run the backward pass for each batch.
        for i in range(grad_output.shape[0]):
            grad_input[i] = grad_output[i] * torch.where(input[i] < 0.0, 0.0, 1)
        
        return grad_input


class CustomSoftmaxLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

        sm = torch.nn.Softmax(dim=dim)
        softmax_output = sm(input)

        ctx.save_for_backward(softmax_output)
        ctx.dim = dim

        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, = ctx.saved_tensors
        dim = ctx.dim
        softmax_output_t = torch.transpose(softmax_output, 0, 1)

        i = torch.eye(softmax_output.shape[1])
        # print(i.shape)
        # print(softmax_output_t.shape)
        # print(grad_output.shape)
        grad_input = grad_output.clone()

        # For each batch, calculate the Jacobian matrix
        for i2,grad_b,softmax_b in zip(range(grad_output.shape[0]),grad_output,softmax_output):
            softmax_t_b = torch.unsqueeze(softmax_b,1) # Transpose softmax_b
            if(softmax_t_b.shape[0] == grad_b.shape[0]): # Check if the dimensions are correct. Note that grad_b has one less dimension, so for both only the first element is looked at

                # subtract softmax from i
                iden_soft = torch.subtract(i, softmax_b)

                # Jacobian matrix fro softmax and iden_soft
                J = torch.matmul(torch.diag(softmax_b),iden_soft)
                grad_input[i2] = torch.matmul(J,grad_b)
            else:
                print("not good")
                J = - torch.matmul(softmax_output_t, softmax_output)
                # print(J.shape)
                grad_input = torch.matmul(grad_output, J)
        return grad_input, None

class CustomConvLayer(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias, stride, kernel_size):
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # implement the cross correlation filter
        
        # weight shape - out_ch x in_ch x kernel_width x kernel_height
        # bias shape - out_ch
        # input shape - batch x ch x width x height
        # out shape - batch x out_ch x width //stride x height //stride
        
        # You can assume the following,
        #  no padding
        #  kernel width == kernel height
        #  stride is identical along both axes

        out_ch = weight.shape[0]
        in_ch = weight.shape[1]

        kernel = kernel_size
        
        batch, _, height, width = input.shape

        # YOUR IMPLEMENTATION HERE!
        output = 1


        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, stride, kernel_size = inputs
        # save the tensors required for back pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride

    @staticmethod
    def backward(ctx, grad_output):
        # grad output shape - batch x out_dim x out_width x out_height (strided)
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        grad_input = grad_weight = grad_bias = None

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        out_ch = weight.shape[0]
        in_ch = weight.shape[1]

        kernel = weight.shape[2]
        
        batch, _, height, width = input.shape

        # YOUR IMPLEMENTATION HERE!

        return grad_input, grad_weight, grad_bias, None, None