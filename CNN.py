import numpy as np

def one_hot_encode(y,num_classes=10):
    return np.eye(num_classes)[y]

def load_csv_data(path, num_classes=10):
    data = np.loadtxt(path, delimiter=',',skiprows=1)
    x = data[:,1:]/255
    y = data[:,0].astype(int)

    x = x.reshape(-1,1,28,28)
    y = one_hot_encode(y,num_classes=num_classes)

    return x,y



class Conv2D:
    def __init__(self,in_channels,out_channels,kernel_size,strides=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides

        # Xavier初始化

        self.weight = np.random.randn(out_channels, in_channels,kernel_size,kernel_size) * np.sqrt(2/in_channels)
        self.bias = np.zeros(out_channels)

    def forward(self,x):
        self.input = x #记录输入用于反向传播
        N, C, H, W = x.shape
        K_h, K_w = self.kernel_size, self.kernel_size
        S = self.strides
        C_out = self.out_channels

        H_out = (H - K_h) // S + 1
        W_out = (W - K_w) // S + 1

        out = np.zeros((N, C_out, H_out, W_out))

        for i in range(N):
            for j in range(C_out):
                for k in range(H_out):
                    for l in range(W_out):
                        h_start = k * S
                        w_start = l * S
                        h_end = h_start + K_h
                        w_end = w_start + K_w

                        region = x[i,:,h_start:h_end,w_start:w_end]
                        out[i,j,k,l] = np.sum(region * self.weight[j])+self.bias[j]
        
        return out

    def backward(self,grad_output,lr):
        N, C_out, H_out, W_out = grad_output.shape
        C_in, K = self.in_channels, self.kernel_size

        grad_weight = np.zeros_like(self.weight)  # shape = (C_out, C_in, K, K)
        grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(self.input)

        for i in range(N):
            for j in range(C_out):
                for y in range(H_out):
                    for x in range(W_out):
                        grad_val = grad_output[i, j, y, x]
                        h0 = y * self.strides
                        w0 = x * self.strides

                        # region.shape == (C_in, K, K)
                        region = self.input[
                            i,
                            :,
                            h0:h0 + K,
                            w0:w0 + K
                        ]

                        # grad_weight[j] 本身也是 (C_in, K, K)
                        grad_weight[j] += grad_val * region
                        grad_input[
                            i,
                            :,
                            h0:h0 + K,
                            w0:w0 + K
                        ] += grad_val * self.weight[j]

                grad_bias[j] += np.sum(grad_output[i, j])

        # 更新参数
        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias

        return grad_input

class ReLU:
    def forward(self,x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self,grad_output):
        return grad_output * self.mask


class MaxPool2d:
    def __init__(self, kernel_size=2, strides=2):
        self.kernel_size = kernel_size
        self.strides = strides

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.strides


        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1
        out = np.zeros((N, C, H_out, W_out))
        self.argmax = np.zeros_like(out, dtype=int)

        for i in range(N):
            for j in range(C):
                for k in range(H_out):
                    for l in range(W_out):
                        h_start = k * S
                        w_start = l * S
                        h_end = h_start + K
                        w_end = w_start + K

                        region = x[i, j, h_start:h_end, w_start:w_end]
                        flat_index = np.argmax(region)
                        self.argmax[i,j,k,l] = flat_index
                        out[i, j, k, l] = region.flatten()[flat_index]
        return out

    def backward(self,grad_output):
        N, C, H_out, W_out = grad_output.shape
        K = self.kernel_size
        S = self.strides
        grad_input = np.zeros_like(self.input)
        for i in range(N):
            for j in range(C):
                for k in range(H_out):
                    for l in range(W_out):
                        h_start = k * S
                        w_start = l * S

                        flat_index = self.argmax[i,j,k,l]
                        h_index = flat_index // K
                        w_index = flat_index % K
                        grad_input[i,j,h_start+h_index,w_start+w_index] = grad_output[i,j,k,l]
        return grad_input



class Flatten:
    def forward(self,x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0],-1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class Dense:
    def __init__(self,in_features,out_features):
        self.weight = np.random.randn(in_features,out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)

    def forward(self,x):
        self.input = x
        return np.dot(x,self.weight) + self.bias

    def backward(self,grad_output,lr):
        grad_input = np.dot(grad_output,self.weight.T)
        grad_weight = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias

        return grad_input

class Softmax:
    def forward(self,x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

class CrossEntropyLoss:
    def forward(self,probs,y_true):
        self.probs = probs
        self.y_true = y_true
        N = probs.shape[0]
        loss = -np.sum(self.y_true * np.log(probs + 1e-12)) / N
        return loss

    def backward(self):
        N = self.probs.shape[0]
        return (self.probs - self.y_true) / N







class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 8, 3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2,2)

        self.conv2 = Conv2D(8, 16, 3)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2,2)

        self.flatten = Flatten()

        self.fc1 = Dense(400, 128)
        self.relu3 = ReLU()
        self.fc2 = Dense(128, 10)
        self.softmax = Softmax()

    def forward(self,x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)

        x = self.fc2.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self,grad,lr):
        grad = self.fc2.backward(grad,lr)
        grad = self.relu3.backward(grad)
        grad = self.fc1.backward(grad,lr)

        grad = self.flatten.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad,lr)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad,lr)
        return grad

def evaluate(model,x,y_true):
    y_pred = model.forward(x)
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = y_true.argmax(axis=1)
    acc = np.mean(pred_labels == true_labels)
    return acc

def iterate_minibatches(x, y, batchsize, shuffle=True):
    N = x.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, N, batchsize):
        end = start_idx + batchsize
        batch_indices = indices[start_idx:end]
        yield x[batch_indices], y[batch_indices]

def train(model,x_train,y_train,x_val=None,y_val=None,epochs=3,batchsize=64,lr=0.01):
    criterion = CrossEntropyLoss()
    for epoch in range(1,epochs + 1):
        running_loss = 0.0
        seen = 0
        for x_batch, y_batch in iterate_minibatches(x_train, y_train, batchsize, shuffle=True):
            probs = model.forward(x_batch)
            loss = criterion.forward(probs,y_batch)
            grad = criterion.backward()
            model.backward(grad,lr)

            running_loss += loss*x_batch.shape[0]
            seen += x_batch.shape[0]

        avg_loss = running_loss / max(1,seen)
        msg = f"epoch: {epoch}, loss: {avg_loss}"
        if x_val is not None and y_val is not None:
            acc = evaluate(model,x_val,y_val)
            msg += f", acc: {acc}"
        print(msg)




x_train,y_train = load_csv_data('fashion-mnist_train.csv')
x_test,y_test = load_csv_data('fashion-mnist_test.csv')

np.random.seed(42)
model = SimpleCNN()

train(model,x_train[:1000],y_train[:1000])

print(evaluate(model,x_test[:1000],y_test[:1000]))





















