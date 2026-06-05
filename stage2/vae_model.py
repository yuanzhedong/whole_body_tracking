"""Conditional motion VAE for BeyondMimic Stage-2 (Fig 7B-i), per Table S6.

Encoder:  z = E(reference-motion obs)            -> latent (dim 32), "motion intent"
Decoder:  a_hat = D(z, proprioceptive obs)        -> action (29 joint targets)
Trained via DAgger to imitate a frozen tracking policy (the teacher); modified ELBO:
    L = || a_hat - a_teacher ||^2  +  beta * KL( q(z) || N(0, I) )

Table S6 defaults: latent 32, enc/dec MLP [2048,1024,512] ELU, KL coef (beta) 0.01.
"""
import torch
import torch.nn as nn


def mlp(in_dim, hidden, out_dim, act=nn.ELU):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class MotionVAE(nn.Module):
    def __init__(self, ref_dim, proprio_dim, act_dim, latent=32,
                 enc=(2048, 1024, 512), dec=(2048, 1024, 512)):
        super().__init__()
        self.latent = latent
        self.ref_dim, self.proprio_dim, self.act_dim = ref_dim, proprio_dim, act_dim
        self.encoder = mlp(ref_dim, list(enc), 2 * latent)        # -> (mu, logvar)
        self.decoder = mlp(latent + proprio_dim, list(dec), act_dim)

    def encode(self, ref):
        mu, logvar = self.encoder(ref).chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, proprio):
        return self.decoder(torch.cat([z, proprio], dim=-1))

    def forward(self, ref, proprio, sample=True):
        mu, logvar = self.encode(ref)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z, proprio), mu, logvar


def kl_divergence(mu, logvar):
    # KL( N(mu, sigma) || N(0, I) ), mean over batch
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
