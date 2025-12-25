import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50
from typing import Optional
from model.mono.depth_net import depth_feature_res
from model.models import *
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.diffusion_policy.model.diffusion.mlp_diffusion_net import PointWiseDiffusionNet
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.unloc import UnlocFeatureExtractor, DepthUncertaintyHead

RAY_STATS = {"min": torch.tensor([0.0]), "max": torch.tensor([20.0])}
def normalize_data(data, stats):
    stats['min'] = stats['min'].to(data.device)
    stats['max'] = stats['max'].to(data.device)
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    return ndata * 2 - 1

def unnormalize_data(ndata, stats):
    stats['min'] = stats['min'].to(ndata.device)
    stats['max'] = stats['max'].to(ndata.device)
    ndata = (ndata + 1) / 2
    return ndata * (stats['max'] - stats['min']) + stats['min']

class DepthPredModels(nn.Module):
    def __init__(self, config, encoder_type="dptv2", decoder_type="f3mlp"):
        super().__init__()
        """
        encoder:
            res50: resnet50
            RMD: 要传入rmd_matrix
            dptv2: DepthAnythingV2的dinoV2
            res50_3D: Res50 3D先验
            res50_RSK Res50 Rsk
        decoder:
            f3mlp: 分类bin
            diffusion: 之前的diffusion
            
        修改encoder以后，要检查_encoder()函数的逻辑，是否能给decoder返回正确的tensor[B, 40, 128]
        """
        self.config = config
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type # f3mlp / diffusion
        
        # encoder
        self._init_encoders()
        self._init_decoders()
        
    def forward(self, func_name, **kwargs):
        # encoder
        if func_name == "encode": # 集成encode
            features = self._encode(**kwargs)
            return features

        elif func_name == "decoder_train":
            return self._decoder_train_get_pred_loss(cond=kwargs["depth_cond"], gt_ray=kwargs["gt_ray"])
        
        elif func_name == "decoder_inference":
            return self._decoder_inference_get_pred(cond=kwargs["depth_cond"], num_samples=kwargs.get("num_samples", 1), return_uncertainty=kwargs.get("return_uncertainty", False))
        else:
            raise NotImplementedError

    def _encode(self, obs_img, rmd_matrix=None):
        if self.encoder_type == "res50":
            features, _ = self.res50_encoder(obs_img)
        elif self.encoder_type == "res50_3D":
            features, _ = self.res50_3D(obs_img)
        elif self.encoder_type == "res50_RSK":
            features, _ = self.res50_RSK(obs_img)
        elif self.encoder_type == "RMD":
            features, _ = self.res50_encoder(obs_img)
            map_features = self.RMD_slide_encoder(rmd_matrix)
            features = self.rmd_depth_cross_encoder(features, map_features)
            return features
        elif self.encoder_type == "dptv2":
            features, _, _ = self.dptv2_encoder(obs_img=obs_img)
        return features
    
    def _decoder_train_get_pred_loss(self, cond, gt_ray):
        if self.decoder_type == "f3mlp":
            return self._forward_mlp_train(cond, gt_ray)
        elif self.decoder_type == "diffusion":
            return self._forward_diffusion_train(cond, gt_ray)
        elif self.decoder_type == "unloc":
            return self._forward_unloc_train(cond, gt_ray)
        
    def _decoder_inference_get_pred(self, cond, num_samples=1, return_uncertainty=False):
        if self.decoder_type == "f3mlp":
            return self._forward_mlp_inference(cond)
        elif self.decoder_type == "diffusion":
            return self._forward_diffusion_inference(cond)
        elif self.decoder_type == "unloc":
            return self._forward_unloc_inference(cond, return_uncertainty)
    def _forward_mlp_train(self, cond, gt_ray):
        pred = self.f3mlp_decoder(cond)
        loss = F.l1_loss(pred, gt_ray)
        return {"pred": pred, "loss": loss} 
    
    def _forward_mlp_inference(self, cond):
        pred = self.f3mlp_decoder(cond)
        return pred

    def _forward_diffusion_train(self, cond, gt_ray):
        device = cond.device
        B = cond.shape[0]
        norm_ray = normalize_data(gt_ray, RAY_STATS)
        
        noise = torch.randn_like(norm_ray)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=device
        ).long()
        noisy_samples = self.noise_scheduler.add_noise(norm_ray, noise, timesteps)
        noise_pred = self.noise_pred_net(sample=noisy_samples, timestep=timesteps, global_cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        return {"pred": noise_pred, "loss": loss}
        
    def _forward_diffusion_inference(self, cond, num_samples=1):

            device = cond.device
            B = cond.shape[0]
            L = 40 
            if num_samples > 1:
                cond = cond.repeat_interleave(num_samples, dim=0)
                current_batch_size = B * num_samples
            else:
                current_batch_size = B

            inference_steps = self.config.get("num_diffusion_iters", 32) 
            self.noise_scheduler.set_timesteps(inference_steps)
            curr_sample = torch.randn((current_batch_size, L), device=device) 
            
            for t in self.noise_scheduler.timesteps:
                timestep_batch = t.unsqueeze(-1).repeat(current_batch_size).to(device)
                noise_pred = self.noise_pred_net(
                    sample=curr_sample, 
                    timestep=timestep_batch, 
                    global_cond=cond
                )
                curr_sample = self.noise_scheduler.step(
                    model_output=noise_pred, 
                    timestep=t, 
                    sample=curr_sample
                ).prev_sample
            final_pred = unnormalize_data(curr_sample, RAY_STATS)
            
            return final_pred
    
    def _forward_unloc_train(self, cond, gt_ray):
        d, b = self.unloc_decoder(cond)
        
        epsilon = 1e-6
        abs_error = torch.abs(d - gt_ray) # |d_i - d_i(s)|
        term_1 = torch.log(b + epsilon) # log(b_i)
        term_2 = abs_error / (b + epsilon) #|d_i - d_i(s)| / b_i
        loss = torch.mean(term_1 + term_2) # 在批次和射线维度上取平均
        mean_abs_error = torch.mean(abs_error)
        mean_uncertainty = torch.mean(b)
        return {"pred": d, "loss": loss}
    
    def _forward_unloc_inference(self, cond, return_uncertainty=False):
        d, b = self.unloc_decoder(cond)
        
        if return_uncertainty:
            return d, b
        else:
            return d
    
    def _init_encoders(self):
        if self.encoder_type == "RMD":
            self.res50_encoder = depth_feature_res()
            self.RMD_slide_encoder = RMDSlidingWindowEncoder()
            self.rmd_depth_cross_encoder = CrossAttentionStack()
        elif self.encoder_type == "dptv2":
            self.dptv2_encoder = UnlocFeatureExtractor(checkpoint_path=self.config["dptv2_ckpt_path"])
        elif self.encoder_type == "res50":
            self.res50_encoder = depth_feature_res()
        elif self.encoder_type == "res50_3D":
            self.res50_3D = depth_feature_res(checkpoint_path=self.config["res50_3D_ckpt_path"])
        elif self.encoder_type == "res50_RSK":
            self.res50_RSK = depth_feature_res(checkpoint_path=self.config["res50_RSK_ckpt_path"])
    def _init_decoders(self):
        if self.decoder_type == "f3mlp":
            self.f3mlp_decoder = F3MlpDecoder()
        elif self.decoder_type == "diffusion":
            # self.noise_pred_net = ConditionalUnet1D(input_dim=1, global_cond_dim=self.config["encoding_size"], down_dims=self.config["down_dims"], kernel_size=3)
            self.noise_pred_net = PointWiseDiffusionNet()
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.config["num_diffusion_iters"],beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon')
        elif self.decoder_type == "unloc":
            self.unloc_decoder = DepthUncertaintyHead()