import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sear import utils
from sear.models.actor import Actor
from sear.models.critic import Critic
from sear.models.decoder import PoolDecoder
from sear.models.encoder import PoolEncoder, TEDClassifier
from sear.models.random_shifts_aug import RandomShiftsAug


class SEARAgent:

    def __init__(self, obs_shape, action_shape, device, lr, latent_dim,
                 feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 reconstruction_loss_coeff, decoder_lr, mask_lr, ted, ted_lr,
                 mask_loss_coeff, detach_critic, detach_mask_decoder,
                 detach_reconstruction_decoder, split_latent, ted_loss_coeff):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.reconstruction_loss_coeff = reconstruction_loss_coeff
        self.mask_loss_coef = mask_loss_coeff
        self.ted_loss_coeff = ted_loss_coeff
        self.detach_critic = detach_critic
        self.detach_mask_decoder = detach_mask_decoder
        self.detach_reconstruction_decoder = detach_reconstruction_decoder
        self.split_latent = split_latent
        self.ted = ted

        # models
        self.encoder = PoolEncoder(obs_shape, repr_dim=latent_dim).to(device)
        decoder_input_dim = int(latent_dim / 2) if split_latent else latent_dim
        self.mask_decoder = PoolDecoder(in_channels=16,
                                        out_channels=obs_shape[0] // 3,
                                        output_act=nn.Sigmoid(),
                                        repr_dim=decoder_input_dim).to(device)
        self.decoder = PoolDecoder(in_channels=16,
                                   out_channels=obs_shape[0],
                                   repr_dim=decoder_input_dim).to(device)
        self.actor = Actor(decoder_input_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(decoder_input_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(decoder_input_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        # self.actor = Actor(self.encoder.repr_dim/2, action_shape, feature_dim,
        #                    hidden_dim).to(device)
        # self.critic = Critic(self.encoder.repr_dim/2, action_shape, feature_dim,
        #                      hidden_dim).to(device)
        # self.critic_target = Critic(self.encoder.repr_dim, action_shape,
        #                             feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.ted:
                # self.ted_classifier = TEDClassifier(self.encoder.repr_dim).to(self.device)
            self.ted_classifier = TEDClassifier(decoder_input_dim).to(self.device)

        # Loss functions
        self.reconstruction_loss_fn = nn.MSELoss(reduction="none")
        self.mask_loss_fn = nn.BCELoss(reduction="none")

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),
                                            lr=decoder_lr)
        self.mask_opt = torch.optim.Adam(self.mask_decoder.parameters(),
                                         lr=mask_lr)
        if self.ted:
            self.ted_optimizer = torch.optim.Adam(self.ted_classifier.parameters(), lr=ted_lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.decoder.train(training)
        self.mask_decoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        if self.ted:
            self.ted_classifier.train(training)

    def act(self, obs, step, eval_mode):
        obs = obs["pixels"]
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        _, f2 = self.get_latent_splits(obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(f2, stddev)
        # dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def get_latent_splits(self, encoded_obs):
        if not self.split_latent:
            return encoded_obs, encoded_obs

        f1 = encoded_obs[:, :int(encoded_obs.shape[1] / 2)]
        f2 = encoded_obs[:, int(encoded_obs.shape[1] / 2):]
        return f1, f2

    def update_representation(self, encoded_obs, encoded_next_obs, encoded_same_episode_obs, obs_shape):
        obs_shape = obs_shape[0]
        # Stack the consecutive observations to make temporal samples
        non_iid_samples = torch.stack([encoded_obs, encoded_next_obs], dim=1)
        # All temporal samples are given a label of 1
        non_iid_labels = torch.ones((obs_shape))

        # Create the non-temporal different episode samples
        rnd_idx = torch.randperm(obs_shape)
        diff_ep_iid_samples = torch.stack([encoded_obs, encoded_next_obs[rnd_idx]], dim=1)
        # All non-temporal samples are given a label of 0
        diff_ep_iid_labels = torch.zeros((obs_shape))

        # Create the non-temporal same episode samples
        same_ep_iid_samples = torch.stack([encoded_obs, encoded_same_episode_obs], dim=1)
        same_ep_iid_labels = torch.zeros((obs_shape))

        samples = torch.cat([non_iid_samples, diff_ep_iid_samples, same_ep_iid_samples])
        labels = torch.cat([non_iid_labels, diff_ep_iid_labels, same_ep_iid_labels]).to(self.device)

        r = self.ted_classifier(samples)
        ted_loss = nn.BCEWithLogitsLoss()
        ted_loss = ted_loss(r, labels)
        # ted_loss = self.ted_loss(r, labels) * self.ted_coef
        # ted_loss.backward()
        # self.ted_optimizer.step()
        return ted_loss

    # Critic and decoder are coupled since both the critic and decoder losses go through the encoder
    def update_critic_and_decoders(self, obs, encoded_obs, action, reward,
                                   discount, encoded_next_obs, step,
                                   robot_masks, encoded_same_episode_obs, obs_shape):
        metrics = {}

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            _, f2_next = self.get_latent_splits(encoded_obs)
            dist = self.actor(f2_next, stddev)
            # dist = self.actor(encoded_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(f2_next,
                                                      next_action)
            # target_q1, target_q2 = self.critic_target(encoded_next_obs,
                                                    #   next_action)
            target_v = torch.min(target_q1, target_q2)
            target_q = reward + (discount * target_v)

        # q1, q2 = self.critic(
        #     encoded_obs.detach() if self.detach_critic else encoded_obs, action)
        # critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        f1, f2 = self.get_latent_splits(encoded_obs)
        reconstructed_obs = self.decoder(
            f1.detach() if self.detach_reconstruction_decoder else f1)
        reconstructed_mask = self.mask_decoder(
            f2.detach() if self.detach_mask_decoder else f2)
        # reconstructed_mask = self.mask_decoder(encoded_obs.detach() if self.detach_mask_decoder else encoded_obs)
        obs = obs / 255.0 - 0.5
        reconstructed_obs = torch.clamp(reconstructed_obs, -0.5, 0.5)

        q1, q2 = self.critic(
            f2.detach() if self.detach_critic else f2, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        reconstruction_loss = self.reconstruction_loss_fn(
            reconstructed_obs, obs)
        reconstruction_loss = (reconstruction_loss.reshape(
            obs.shape[0], -1).sum(dim=1).mean())

        mask_loss = self.mask_loss_fn(reconstructed_mask, robot_masks)
        mask_loss = mask_loss.reshape(robot_masks.shape[0],
                                      -1).sum(dim=1).mean()
        
        _,f2_same_episode = self.get_latent_splits(encoded_same_episode_obs)
        ted_loss = self.update_representation(f2, f2_next, f2_same_episode, obs_shape)
        # ted_loss = self.update_representation(encoded_obs, encoded_next_obs, encoded_same_episode_obs, obs_shape)
        loss = critic_loss + reconstruction_loss * self.reconstruction_loss_coeff + mask_loss * self.mask_loss_coef + ted_loss * self.ted_loss_coeff
        # loss = critic_loss + ted_loss * self.ted_loss_coeff
        self.ted_optimizer.zero_grad()
        self.encoder_opt.zero_grad(set_to_none=True)
        self.decoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.mask_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        self.decoder_opt.step()
        self.mask_opt.step()
        self.ted_optimizer.step()

        if self.use_tb:
            metrics["critic_target_q"] = target_q.mean().item()
            metrics["critic_q1"] = q1.mean().item()
            metrics["critic_q2"] = q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()
            metrics["reconstruction_loss"] = reconstruction_loss.item()
            metrics["mask_loss"] = mask_loss.item()

        return metrics

    def update_actor(self, encoded_obs, step):
        metrics = {}

        stddev = utils.schedule(self.stddev_schedule, step)
        _, f2 = self.get_latent_splits(encoded_obs)
        dist = self.actor(f2, stddev)
        # dist = self.actor(encoded_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        q1, q2 = self.critic(f2, action)
        # q1, q2 = self.critic(encoded_obs, action)
        q = torch.min(q1, q2)

        actor_loss = -q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics
         
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, same_episode_obs = batch
        obs_shape = obs["pixels"].shape
        rgb_obs = obs["pixels"]
        next_rgb_obs = next_obs["pixels"]
        rgb_same_episode_obs = same_episode_obs["pixels"]

        masks = obs["segmentation"]
        rgb_obs, action, reward, discount, next_rgb_obs, masks, rgb_same_episode_obs = utils.to_torch(
            (rgb_obs, action, reward, discount, next_rgb_obs, masks, rgb_same_episode_obs),
            self.device)

        # augment
        rgb_obs = self.aug(rgb_obs.float())
        masks = self.aug(masks.float(), repeat_last=True)
        next_rgb_obs = self.aug(next_rgb_obs.float())
        rgb_same_episode_obs = self.aug(rgb_same_episode_obs.float())
        # encode
        encoded_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_rgb_obs)
            encoded_same_episode_obs = self.encoder(rgb_same_episode_obs)
        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic and decoder
        metrics.update(
            self.update_critic_and_decoders(rgb_obs, encoded_obs, action,
                                            reward, discount, encoded_next_obs,
                                            step, masks, encoded_same_episode_obs, obs_shape))

        # update actor
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def get_frames_to_record(self, obs):
        rgb_obs = obs["pixels"]
        rgb_obs_torch, = utils.to_torch((rgb_obs,), self.device)

        with torch.no_grad():
            encoded_obs = self.encoder(rgb_obs_torch.unsqueeze(0))
            f1, f2 = self.get_latent_splits(encoded_obs)
            reconstructed_obs = self.decoder(f1).detach().cpu().numpy()
            reconstructed_mask = self.mask_decoder(f2).detach().cpu().numpy()

        reconstructed_obs = (reconstructed_obs[0] + 0.5) * 255.0
        reconstructed_mask = reconstructed_mask[0] * 255.0
        masks = obs["segmentation"] * 255.0

        frame_names = ["rgb", "mask", "reconstructed_rgb", "reconstructed_mask"]
        frames = {}
        for k, v in zip(
                frame_names,
            [rgb_obs, masks, reconstructed_obs, reconstructed_mask]):
            frames[k] = v[-3:].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

        return frames

    def load_pretrained_weights(self, pretrain_path, just_encoder_decoders):
        if just_encoder_decoders:
            print("Loading pretrained encoder and decoders")
        else:
            print("Loading entire agent")

        payload = torch.load(pretrain_path, map_location="cpu")
        pretrained_agent = payload['agent']

        self.encoder.load_state_dict(pretrained_agent.encoder.state_dict())
        self.decoder.load_state_dict(pretrained_agent.decoder.state_dict())
        self.mask_decoder.load_state_dict(
            pretrained_agent.mask_decoder.state_dict())

        if not just_encoder_decoders:
            self.actor.load_state_dict(pretrained_agent.actor.state_dict())
            self.critic.load_state_dict(pretrained_agent.critic.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
