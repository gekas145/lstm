import torch
import numpy as np
from torch.nn.functional import sigmoid, tanh



class LSTM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, W, R, W_proj, p, b, h_init=None, c_init=None):

        batch, time, _ = inputs.shape
        out_dim = p.shape[0] // 3
        proj_dim = W_proj.shape[1]

        gates = torch.zeros((4, batch, time, out_dim))
        h = torch.zeros((batch, time, out_dim))
        h_proj = torch.zeros((batch, time + 1, proj_dim))
        c = torch.zeros((batch, time + 1, out_dim))

        if h_init is not None:
            h_proj[:, 0, :] = h_init
        if c_init is not None:
            c[:, 0, :] = c_init

        for t in range(time):
            gates[0, :, t, :] = sigmoid(inputs[:, t, :] @ W[:, :out_dim] +\
                                        h_proj[:, t, :] @ R[:, :out_dim] +\
                                        c[:, t, :] * p[:out_dim] +\
                                        b[:out_dim])
            
            gates[1, :, t, :] = sigmoid(inputs[:, t, :] @ W[:, out_dim:2*out_dim] +\
                                        h_proj[:, t, :] @ R[:, out_dim:2*out_dim] +\
                                        c[:, t, :] * p[out_dim:2*out_dim] +\
                                        b[out_dim:2*out_dim])
            
            gates[2, :, t, :] = tanh(inputs[:, t, :] @ W[:, 2*out_dim:3*out_dim] +\
                                     h_proj[:, t, :] @ R[:, 2*out_dim:3*out_dim] +\
                                     b[2*out_dim:3*out_dim])
            
            c[:, t+1, :] = gates[1, :, t, :] * c[:, t, :] + gates[0, :, t, :] * gates[2, :, t, :]

            gates[3, :, t, :] = sigmoid(inputs[:, t, :] @ W[:, 3*out_dim:4*out_dim] +\
                                        h_proj[:, t, :] @ R[:, 3*out_dim:4*out_dim] +\
                                        c[:, t + 1, :] * p[2*out_dim:3*out_dim] +\
                                        b[3*out_dim:4*out_dim])
            
            h[:, t, :] = gates[3, :, t, :] * tanh(c[:, t+1, :])
            h_proj[:, t+1, :] = h[:, t, :] @ W_proj

        ctx.save_for_backward(inputs, W, R, W_proj, p, b, h_proj, h, c, gates)

        return h_proj[:, 1:, :]

    @staticmethod
    def backward(ctx, out_grad):
        inputs, W, R, W_proj, p, b, h_proj, h, c, gates = ctx.saved_tensors

        batch, time, _ = inputs.shape
        out_dim = p.shape[0] // 3

        gates_grad = torch.zeros((4, batch, out_dim))

        W_grad = torch.zeros_like(W)
        R_grad = torch.zeros_like(R)
        W_proj_grad = torch.zeros_like(W_proj)
        p_grad = torch.zeros_like(p)
        b_grad = torch.zeros_like(b)
        if inputs.requires_grad:
            inputs_grad = torch.zeros_like(inputs)
        else:
            inputs_grad = None

        h_proj_grad = out_grad[:, -1, :]
        h_grad = h_proj_grad @ W_proj.T
        c_tanh = tanh(c[:, -1, :])
        gates_grad[3, ...] = h_grad * c_tanh * gates[3, :, -1, :] * (1 - gates[3, :, -1, :])
        c_grad = gates_grad[3, ...] * p[2*out_dim:3*out_dim] + h_grad * (1 - c_tanh**2) * gates[3, :, -1, :]

        for t in range(-1, -time-1, -1):

            gates_grad[0, ...] = c_grad * gates[2, :, t, :] * gates[0, :, t, :] * (1 - gates[0, :, t, :])
            gates_grad[1, ...] = c_grad * c[:, t-1, :] * gates[1, :, t, :] * (1 - gates[1, :, t, :])
            gates_grad[2, ...] = c_grad * gates[0, :, t, :] * (1 - gates[2, :, t, :]**2)

            for k in range(4):
                R_grad[:, k*out_dim:(k+1)*out_dim] += h_proj[:, t-1, :].T @ gates_grad[k, ...]
                W_grad[:, k*out_dim:(k+1)*out_dim] += inputs[:, t, :].T @ gates_grad[k, ...]
                b_grad[k*out_dim:(k+1)*out_dim] += torch.sum(gates_grad[k, ...], axis=0)

                if k > 2:
                    break

                step, p_gate_num = (t-1, k) if k <= 1 else (t, k+1)
                p_grad[k*out_dim:(k+1)*out_dim] += torch.sum(c[:, step, :] * gates_grad[p_gate_num, ...], axis=0)

            W_proj_grad += h[:, t, :].T @ h_proj_grad

            if inputs.requires_grad:
                inputs_grad[:, t, :] = gates_grad[0, ...] @ W[:, :out_dim].T +\
                                       gates_grad[1, ...] @ W[:, out_dim:2*out_dim].T +\
                                       gates_grad[2, ...] @ W[:, 2*out_dim:3*out_dim].T +\
                                       gates_grad[3, ...] @ W[:, 3*out_dim:4*out_dim].T

            if t == -time:
                break
            
            h_proj_grad = gates_grad[0, ...] @ R[:, :out_dim].T +\
                          gates_grad[1, ...] @ R[:, out_dim:2*out_dim].T +\
                          gates_grad[2, ...] @ R[:, 2*out_dim:3*out_dim].T +\
                          gates_grad[3, ...] @ R[:, 3*out_dim:4*out_dim].T +\
                          out_grad[:, t-1, :]
            
            h_grad = h_proj_grad @ W_proj.T
            
            c_tanh = tanh(c[:, t-1, :])
            
            gates_grad[3, ...] = h_grad * c_tanh * gates[3, :, t-1, :] * (1 - gates[3, :, t-1, :])

            c_grad = gates_grad[0, ...] * p[:out_dim] +\
                     gates_grad[1, ...] * p[out_dim:2*out_dim] +\
                     c_grad * gates[1, :, t, :] +\
                     gates_grad[3, ...] * p[2*out_dim:3*out_dim] +\
                     h_grad * gates[3, :, t-1, :] * (1 - c_tanh**2)
            
        return inputs_grad, W_grad, R_grad, W_proj_grad, p_grad, b_grad, None, None
    

class SimpleNet(torch.nn.Module):

    def __init__(self, input_dim, lstm_dim, output_dim, *args, proj_dim=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.lstm_dim = lstm_dim
        self.proj_dim = lstm_dim if proj_dim is None else proj_dim

        self.W1 = self.__uniform(input_dim, 4*lstm_dim)
        self.R1 = self.__uniform(self.proj_dim, 4*lstm_dim)
        self.W_proj1 = self.__uniform(lstm_dim, self.proj_dim)
        self.p1 = self.__uniform(3*lstm_dim)
        self.b1 = self.__uniform(4*lstm_dim)

        self.linear = torch.nn.Linear(self.proj_dim, output_dim)

    def forward(self, inputs):
        outputs = LSTM.apply(inputs, self.W1, self.R1, self.W_proj1, self.p1, self.b1)
        return self.linear(outputs)

    def __uniform(self, *size):
        return torch.nn.parameter.Parameter((2 * torch.rand(size) - 1)/np.sqrt(self.lstm_dim))




