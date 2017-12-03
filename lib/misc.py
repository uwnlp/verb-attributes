import torch


def _normalize(input_data, eps=1e-8):
    input_data_denom = torch.sqrt(torch.sum(torch.pow(input_data, 2), 1)).clamp(min=1e-8)
    normed = input_data / input_data_denom.expand_as(input_data)
    return normed


def cosine_ranking_loss(input_data, ctx, margin=0.1):
    """
    :param input_data: [batch_size, 300] tensor of predictions
    :param ctx: [batch_size, 300] tensor of ground truths
    :param margin: Difference between them
    :return: 
    """
    normed = _normalize(input_data)
    ctx_normed = _normalize(ctx)
    shuff_inds = torch.randperm(normed.size(0))
    if ctx.is_cuda:
        shuff_inds = shuff_inds.cuda()
    shuff = ctx_normed[shuff_inds]

    correct_contrib = torch.sum(normed * ctx_normed, 1).squeeze()
    incorrect_contrib = torch.sum(normed * shuff, 1).squeeze()

    # similarity = torch.mm(normed, ctx_normed.t()) #[predictions, gts]
    # correct_contrib = similarity.diag()
    # incorrect_contrib = incorrect_contrib.sum(1).squeeze()/(incorrect_contrib.size(1)-1.0)
    #
    cost = (0.1 + incorrect_contrib-correct_contrib).clamp(min=0)

    return cost, correct_contrib, incorrect_contrib


class CosineRankingLoss(torch.nn.Module):
    def __init__(self, margin=0.1, size_average=True):
        super(CosineRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input_data, ctx):
        cost, _, _ = cosine_ranking_loss(input_data, ctx, self.margin)
        if self.size_average:
            return torch.mean(cost)
        else:
            return torch.sum(cost)

def get_cosine_ranking(input_data, all_words, labels):
    """
    Get the ranking of our predictions
    :param input_data: [batch_size, 300] inputs 
    :param all_words: [dict_size, 300] all possible candidates
    :param labels: [batch_size] array of indices of the GTs
    :return: 
    """
    normed = _normalize(input_data)
    all_words_normed = _normalize(all_words)

    similarity = torch.mm(normed, all_words_normed.t()) #[batch_size, dict_size]
    return get_ranking(similarity, labels)

def get_ranking(predictions, labels, num_guesses=5):
    """
    Given a matrix of predictions and labels for the correct ones, get the number of guesses
    required to get the prediction right per example.
    :param predictions: [batch_size, range_size] predictions
    :param labels: [batch_size] array of labels
    :param num_guesses: Number of guesses to return
    :return: 
    """
    assert labels.size(0) == predictions.size(0)
    assert labels.dim() == 1
    assert predictions.dim() == 2

    values, full_guesses = predictions.topk(predictions.size(1), dim=1)
    _, ranking = full_guesses.topk(full_guesses.size(1), dim=1, largest=False)
    gt_ranks = torch.gather(ranking.data, 1, labels[:, None]).squeeze()

    guesses = full_guesses[:, :num_guesses]
    return gt_ranks, guesses



def optimize(f):
    """
    Decorator for an optimize loop
    """
    def optimize_loop_wrapper(*args, **kwargs):
        if not 'optimizers' in kwargs:
            raise ValueError("When using @optimize, must pass in list of optimizers")
        for opt in kwargs['optimizers']:
            opt.zero_grad()
        loss = f(*args, **kwargs)
        loss.backward()
        for opt in kwargs['optimizers']:
            opt.step()
        return loss.data[0]
    return optimize_loop_wrapper


def print_para(opt):
    """
    Prints parameters of a model
    :param opt: 
    :return: 
    """
    st = []
    for p_name, p in opt.named_parameters():
        st.append("({}) {}: {}".format('grad' if p.requires_grad else '    ',
                                       p_name, p.size()))
    return '\n'.join(st)