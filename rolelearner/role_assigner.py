import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class RoleAssignmentLSTM(nn.Module):
    def __init__(
            self,
            num_roles,
            filler_embedding,
            hidden_dim,
            role_embedding_dim,
            num_layers=1,
            role_assignment_shrink_filler_dim=None,
            bidirectional=False,
            softmax_roles=False
    ):
        super(RoleAssignmentLSTM, self).__init__()
        # TODO: when we move to language models, we will need to use pre-trained word embeddings.
        # See embedder_squeeze in TensorProductEncoder

        self.snap_one_hot_predictions = False

        self.filler_embedding = filler_embedding
        filler_embedding_dim = filler_embedding.embedding_dim

        self.shrink_filler = False
        if role_assignment_shrink_filler_dim:
            self.shrink_filler = True
            self.filler_shrink_layer = nn.Linear(filler_embedding.embedding_dim,
                                                 role_assignment_shrink_filler_dim)
            filler_embedding_dim = role_assignment_shrink_filler_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_roles = num_roles
        self.bidirectional = bidirectional

        # OPTION we may want the LSTM to be bidirectional for things like RTL roles.
        # Also, should the output size be the number of roles for the weight vector?
        # Or is the output of variable size and we apply a linear transformation
        # to get the weight vector?
        self.lstm = nn.LSTM(filler_embedding_dim, hidden_dim, bidirectional=bidirectional,
                            num_layers=self.num_layers)
        if bidirectional:
            print("The role assignment LSTM is bidirectional")
            self.role_weight_predictions = nn.Linear(hidden_dim * 2, num_roles)
        else:
            self.role_weight_predictions = nn.Linear(hidden_dim, num_roles)

        self.softmax_roles = softmax_roles
        if softmax_roles:
            print("Use softmax for role predictions")
            # The output of role_weight_predictions is shape
            # (sequence_length, batch_size, num_roles)
            # We want to softmax across the roles so set dim=2
            self.softmax = nn.Softmax(dim=2)

        self.role_embedding = nn.Embedding(num_roles, role_embedding_dim)
        self.role_indices = torch.tensor([x for x in range(num_roles)], device=device)

    def forward(self, filler_tensor):
        """
        :param filler_tensor: This input tensor should be of shape (batch_size, sequence_length)
        :param filler_lengths: A list of the length of each sequence in the batch. This is used
            for padding the sequences.
        :return: A tensor of size (sequence_length, batch_size, role_embedding_dim) with the role
            embeddings for the input filler_tensor.
        """
        batch_size = len(filler_tensor)
        hidden = self.init_hidden(batch_size)

        fillers_embedded = self.filler_embedding(filler_tensor)
        if self.shrink_filler:
            fillers_embedded = self.filler_shrink_layer(fillers_embedded)
        # The shape of fillers_embedded should be
        # (batch_size, sequence_length, filler_embedding_dim)
        # Pytorch LSTM expects data in the shape (sequence_length, batch_size, feature_dim)
        fillers_embedded = torch.transpose(fillers_embedded, 0, 1)

        '''
        fillers_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            fillers_embedded,
            filler_lengths,
            batch_first=False
        )

        lstm_out, hidden = self.lstm(fillers_embedded, hidden)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        '''
        lstm_out, hidden = self.lstm(fillers_embedded, hidden)
        role_predictions = self.role_weight_predictions(lstm_out)

        if self.softmax_roles:
            role_predictions = self.softmax(role_predictions)
        # role_predictions is size (sequence_length, batch_size, num_roles)

        role_embeddings = self.role_embedding(self.role_indices)

        # Normalize the embeddings. This is important so that role attention is not overruled by
        # embeddings with different orders of magnitude.
        role_embeddings = role_embeddings / torch.norm(role_embeddings, dim=1).unsqueeze(1)
        # role_embeddings is size (num_roles, role_embedding_dim)

        # During evaluation, we want to snap the role predictions to a one-hot vector
        if self.snap_one_hot_predictions:
            one_hot_predictions = self.one_hot_embedding(torch.argmax(role_predictions, 2),
                                                        self.num_roles)
            roles = torch.matmul(one_hot_predictions, role_embeddings)
        else:
            roles = torch.matmul(role_predictions, role_embeddings)
        # roles is size (sequence_length, batch_size, role_embedding_dim)

        return roles, role_predictions

    def init_hidden(self, batch_size):
        layer_multiplier = 1
        if self.bidirectional:
            layer_multiplier = 2

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        # We need a tuple for the hidden state and the cell state of the LSTM.
        return (torch.zeros(self.num_layers * layer_multiplier, batch_size, self.hidden_dim,
                            device=device),
                torch.zeros(self.num_layers * layer_multiplier, batch_size, self.hidden_dim,
                            device=device))

    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes, device=device)
        return y[labels]


if __name__ == "__main__":
    import torch

    num_roles = 10
    filler_embedding_dim = 20
    num_fillers = 10
    hidden_dim = 30
    role_embedding_dim = 20
    filler_embedding = torch.nn.Embedding(
        num_fillers + 1,
        filler_embedding_dim,
        padding_idx=num_fillers
    )

    lstm = RoleAssignmentLSTM(
        num_roles,
        filler_embedding,
        hidden_dim,
        role_embedding_dim,
        num_layers=2,
        bidirectional=True,
    )

    #data = [[1, 2, 3, 4], [1, 8, 1, 0]]
    data = [[2, 3], [1, 10]]
    data_tensor = torch.tensor(data)
    out = lstm(data_tensor, [2, 1])
    print(out)
    print('experiment 2')
    data2 = [[1]]
    data_tensor2 = torch.tensor(data2)
    out2 = lstm(data_tensor2, [1])
    print(out2)
