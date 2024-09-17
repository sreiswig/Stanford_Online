import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###

        # Each set of labels and images are concatenated together
        # 1. Concatenate the full(support and query) set of labels and images
        examples = torch.cat([input_images, input_labels], dim=-1)
        B, K_1, N, D = examples.shape

        
        reshaped_examples = examples.reshape(B, -1, D)
        # Create Support Set and Query Set examples
        support_set_examples = reshaped_examples[:, :-N, :]
        query_set_examples = reshaped_examples[:, -N:, :]

        # 2. Zero out the labels from the concatenated corresponding to the query set
        real_test_labels = query_set_examples[:, :, 784:] 
        zero_labels = torch.zeros_like(real_test_labels)
        query_set_examples[:, :, 784:] = zero_labels

        # lstm input = tensor of shape (N, L, H_in) when batch_first = true
        # N = batch size == B
        # L = sequence length == N*K Feed K labeled examples of each of N classes
        # H_in = input_size == num_classes + 784
        reshaped_examples = torch.cat([support_set_examples, query_set_examples], dim=1)
        # 3. Pass the concatenated set sequentially to the memory-augmented network
        outputs, _ = self.layer1(reshaped_examples)
        outputs, _ = self.layer2(outputs)

        # Output of shape N == B 
        # L == N*K
        # D*H_out == 1 * hidden_size
        # when batch_first = true
        return outputs.reshape(B, K_1, N, N)
        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ###
        B, K_1, N, D = preds.shape
        test_preds = preds[:, -1, :, :] # (B, N, N)
        test_labels = labels[:, -1, :, :]  # (B, N, N)
        test_preds = F.log_softmax(test_preds, dim=-1)
        test_preds = test_preds.view(B, -1)
        test_labels = test_labels.view(B, -1)

        # Assuming test_preds are logits (unnormalized scores)
        loss = F.cross_entropy(test_preds, test_labels, reduction='mean')
        ### END CODE HERE ###

        return loss
