def outside(model, A, Z, X, length):
    B = {(0, length): np.array([[1.0]])}
    for diff in reversed(xrange(1, length + 1)):
        for start in xrange(length + 1 - diff):
            end = start + diff
            span = (start, end)
            for split in xrange(start + 1, end):
                left_span, right_span = ((start, split), (split, end))
                B[left_span] = B[span] * Z[span][split].data * A[right_span][0]
                B[right_span] = B[span] * Z[span][split].data * A[left_span][0]
    return B


def mu(A, B, length):
    Mu = {}
    for index in xrange(0, length):
        span = (index, index + 1)
        split = index

        if span not in Mu:
            Mu[span] = {}

        Mu[span][0] = A[span][0] * B[span][0]

    for diff in xrange(2, length + 1):
        for start in xrange(length + 1 - diff):
            end = start + diff
            span = (start, end)

            if span not in Mu:
                Mu[span] = {}

            Mu[span][0] = A[span][0] * B[span][0]
            for split in xrange(start + 1, end):
                Mu[span][split] = A[span][split] * B[span][0]
    return Mu


def EStep(model, sent, length, sentence_grammar):
    Z, X = calcZ(model, sent, length, train=False)
    A = inside(model, sent, length, sentence_grammar, Z, train=True)
    return A


def MStep(model, Z, Mu, A, length):
        loss = 0
        for index in xrange(0, length):
            span = (index, index + 1)
            split = index

            loss += (-F.log(Z[span][0]) + F.log(model.z_leaf)) * Mu[span][0]

        for diff in xrange(2, length + 1):
            for start in xrange(0, length + 1 - diff):
                end = start + diff
                span = (start, end)

                for split in xrange(start + 1, end):
                    loss += -F.log(Z[span][split]) * Mu[span][split]

        return loss/A[(0, length)][0]


