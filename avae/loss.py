import logging

import torch


class AffinityLoss:
    """Affinity loss based on pre-calculated shape similarity.

    Parameters
    ----------
    lookup : np.ndarray (M, M)
        A square symmetric matrix where each column and row is the index of an
        object from the training set, consisting of M different objects. The
        value at (i, j) is a scalar value encoding the shape similarity between
        objects i and j, pre-calculated using some shape (or other) metric. The
        identity of the matrix should be 1 since these objects are the same
        shape. The affinity similarity should be normalized to the range
        (-1, 1).

    Notes
    -----
    The final loss is calculated using L1-norm. This could be changed, e.g.
    L2-norm. Not sure what the best one is yet.
    """

    def __init__(self, lookup: torch.Tensor, device: torch.device):
        self.device = device
        self.lookup = torch.tensor(lookup).to(device)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = torch.nn.L1Loss()

    def __call__(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Return the affinity loss.

        Parameters
        ----------
        y_true : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing
            the identity of the object as an index. These indices should
            correspond to the rows and columns of the `lookup` table.
        y_pred : torch.Tensor (N, latent_dims)
            An array of latent encodings of the N objects.

        Returns
        -------
        loss : torch.Tensor
            The affinity loss.
        """
        # first calculate the affinity, for the real classes
        c = (
            torch.combinations(y_true, r=2, with_replacement=False)
            .to(self.device)
            .long()
        )
        affinity = self.lookup[c[:, 0], c[:, 1]].to(self.device)

        # now calculate the latent similarity
        z_id = torch.tensor(list(range(y_pred.shape[0])))
        c = torch.combinations(z_id, r=2, with_replacement=False)
        latent_similarity = self.cos(y_pred[c[:, 0], :], y_pred[c[:, 1], :])
        loss = self.l1loss(latent_similarity, affinity)
        return loss


class AVAELoss:
    """AffinityVAE loss consisting of reconstruction loss, beta
    parametrised latent regularisation loss and
    gamma parametrised affinity regularisation loss. Reconstruction loss
    should be Mean Squared Error for real
    valued data and Binary Cross-Entropy for binary data. Latent
    regularisation loss is KL Divergence. Affinity
    regularisation is defined in AffinityLoss class.

    Parameters
    ----------
    device : torch.device
        Device used to calculate the loss.
    beta: list
        Beta parameter list defining weight on latent regularisation term.
    gamma : list
        Gamma parameter list defining weight on affinity regularisation term,
        default = 1. Only used if lookup_aff is present.
    lookup_aff : np.ndarray [M, M]
        A square symmetric matrix where each column and row is the index of an
        object class from the training set, consisting of M different classes.
    recon_loss : 'MSE' or 'BCE'
        Function used for reconstruction loss. BCE uses Binary
        Cross-Entropy for binary data and MSE uses Mean
        Squared Error for real-valued data.

    """

    def __init__(
        self,
        device: torch.device,
        beta: list[float],
        gamma: list[float],
        lookup_aff: torch.Tensor | None = None,
        recon_loss: str = "MSE",
        klreduction: str = "mean",
    ):
        self.device = device
        self.recon_loss = recon_loss
        self.klreduction = klreduction
        self.beta = beta

        self.affinity_loss = None
        self.gamma = gamma

        if lookup_aff is not None and max(gamma) != 0:
            self.affinity_loss = AffinityLoss(lookup_aff, device)

        elif lookup_aff is None and max(gamma) != 0:
            raise RuntimeError(
                "Affinity matrix is needed to compute Affinity loss"
                ". Although you've set gamma, you have not provided --af/"
                "--affinity parameter."
            )
        elif lookup_aff is not None and max(gamma) == 0:
            logging.warning(
                "\n\nWARNING: You provided affinity matrix but no gamma. Unless "
                "you provide gamma, affinity will be ignored and you're "
                "running a vanilla beta-VAE.\n"
            )
            self.affinity_loss = None

    def __call__(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        epoch: int,
        batch_aff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the aVAE loss.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        recon_x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        mu : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent means, where N
            stands for the number of samples in the
            mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        logvar : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent log of the
            variance, where N stands for the number of
            samples in the mini-batch and 'latent_dims' defines the number of
            latent dimensions.
        batch_aff : torch.Tensor (N, )
            Optional, must be present if AVAELoss was
            initialised with 'lookup_aff' parameter containing affinity
            lookup matrix. A vector of N objects in the mini-batch of the
            indices representing the identity of the object's class as
            an index. These indices should correspond to the rows and columns
            of the `lookup_aff` table.

        Returns
        -------
        total_loss : torch.Tensor
            Combined reconstruction, latent regularisaton and affinity loss.
        recon_loss : torch.Tensor
            Reconstruction loss.
        kldivergence : torch.Tensor
            Non-weighted KL Divergence loss.
        affin_loss : torch.Tensor
            Non-weighted affinity loss.
        """
        if self.affinity_loss is not None and batch_aff is None:
            raise RuntimeError(
                "aVAE loss function requires affinity ids for the batch."
            )

        # recon loss
        if self.recon_loss == "BCE":
            recon_loss = torch.nn.functional.binary_cross_entropy(
                x, recon_x, reduction="mean"
            )
        elif self.recon_loss == "MSE":
            recon_loss = torch.nn.functional.mse_loss(
                x, recon_x, reduction="mean"
            )
        else:
            raise RuntimeError(
                "AffinityVAE loss requires 'BCE' or 'MSE' for 'loss_fn' "
                "parameter."
            )

        # kldiv loss
        if self.klreduction == "mean":
            kldivergence = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
        elif self.klreduction == "sum":
            kldivergence = torch.mean(
                -0.5
                * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1),
                axis=0,
            )
        else:
            raise RuntimeError(
                "AffinityVAE loss requires 'mean' or 'sum' for 'klreduction' "
                "parameter."
            )

        # affinity loss
        affin_loss = torch.Tensor([0]).to(self.device)
        if self.affinity_loss is not None:
            affin_loss = self.affinity_loss(batch_aff, mu)

        # total loss
        total_loss = (
            recon_loss
            + self.beta[epoch] * kldivergence
            + self.gamma[epoch] * affin_loss
        )

        return total_loss, recon_loss, kldivergence, affin_loss
