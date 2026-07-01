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


class DualHeadVAE(nn.Module):
    """Control VAE with an auxiliary motion-reconstruction head.

    z = E(ref_window)
      ├─ action head : D(z, proprio) -> a   (BC/DAgger target from BFM-Zero)
      └─ motion head : M(z)          -> ref  (reconstruct the encoder input)

    The motion head forces z to encode the reference motion itself (teacher-independent),
    rather than only the small action-residual the proprio-decoder can't supply. This is
    the fix for the offline latent collapse (EXPERIMENTS: action-only supervision -> z
    unused offline) and gives a generative-ready motion latent for downstream diffusion.
    """

    def __init__(self, ref_dim, proprio_dim, act_dim, latent=32,
                 enc=(2048, 1024, 512), dec=(2048, 1024, 512), mdec=(512, 1024, 2048)):
        super().__init__()
        self.latent = latent
        self.ref_dim, self.proprio_dim, self.act_dim = ref_dim, proprio_dim, act_dim
        self.encoder = mlp(ref_dim, list(enc), 2 * latent)
        self.action_decoder = mlp(latent + proprio_dim, list(dec), act_dim)
        self.motion_decoder = mlp(latent, list(mdec), ref_dim)     # z -> reconstruct ref window

    def encode(self, ref):
        mu, logvar = self.encoder(ref).chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_action(self, z, proprio):
        return self.action_decoder(torch.cat([z, proprio], dim=-1))

    def decode_motion(self, z):
        return self.motion_decoder(z)

    def forward(self, ref, proprio, sample=True):
        mu, logvar = self.encode(ref)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode_action(z, proprio), self.decode_motion(z), mu, logvar
