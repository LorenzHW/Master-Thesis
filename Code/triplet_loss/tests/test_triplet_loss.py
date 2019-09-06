import unittest

import numpy as np

from triplet_loss.triplet_loss import batch_hard_triplet_loss


def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
        pairwise_distances.diagonal())
    return pairwise_distances


class TripletLossTest(unittest.TestCase):
    def test_batch_hard_triplet_loss(self):
        """Test the triplet loss with batch hard triplet mining"""
        num_data = 50
        feat_dim = 6
        margin = 0.2
        num_classes = 5

        embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

        for squared in [True, False]:
            pdist_matrix = pairwise_distance_np(embeddings, squared=squared)

            loss_np = 0.0
            for i in range(num_data):
                # Select the hardest positive
                max_pos_dist = np.max(pdist_matrix[i][labels == labels[i]])

                # Select the hardest negative
                min_neg_dist = np.min(pdist_matrix[i][labels != labels[i]])

                loss = np.maximum(0.0, max_pos_dist - min_neg_dist + margin)
                loss_np += loss

            loss_np /= num_data

            # Compute the loss in TF.
            loss_tf = batch_hard_triplet_loss(labels, embeddings, margin, squared=squared)
            assert np.allclose(loss_np, loss_tf)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TripletLossTest("test_batch_hard_triplet_loss"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
