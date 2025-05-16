# consistency model 
# Karras Scheduler from openAI consistency model
import torch
import numpy as np
from typing import Optional, Tuple, Union
from diffusion.util import randn_tensor

class KarrasScheduler():
    """
    Args:
        sigma_min (`float`): minimum noise magnitude
        sigma_max (`float`): maximum noise magnitude
        s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
            A reasonable range is [1.000, 1.011].
        s_churn (`float`): the parameter controlling the overall amount of stochasticity.
            A reasonable range is [0, 100].
        s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
            A reasonable range is [0, 10].
        s_max (`float`): the end value of the sigma range where we add noise.
            A reasonable range is [0.2, 80].
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho=7.0,
        sigma_data: float = 0.5,
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.num_timesteps = 40

        # setable values
        self.num_inference_steps: int = None
        self.timesteps: np.IntTensor = None
        self.schedule: torch.FloatTensor = None  # sigma(t_i)

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)
        schedule = [
            (
                self.config.sigma_max**2
                * (self.config.sigma_min**2 / self.config.sigma_max**2) ** (i / (num_inference_steps - 1))
            )
            for i in self.timesteps
        ]
        self.schedule = torch.tensor(schedule, dtype=torch.float32, device=device)

    def add_noise_to_input(
        self, sample: torch.FloatTensor, sigma: float, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.FloatTensor, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.
        TODO Args:
        """
        if self.config.s_min <= sigma <= self.config.s_max:
            gamma = min(self.config.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        eps = self.config.s_noise * randn_tensor(sample.shape, generator=generator).to(sample.device)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class
            KarrasVeOutput: updated sample in the diffusion chain and derivative (TODO double check).
        Returns:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] or `tuple`:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """

        pred_original_sample = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - pred_original_sample) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        if not return_dict:
            return (sample_prev, derivative)

        return KarrasVeOutput(
            prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
        )

    def step_correct(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.FloatTensor,
        sample_prev: torch.FloatTensor,
        derivative: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[KarrasVeOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. TODO complete description
        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            sample_prev (`torch.FloatTensor`): TODO
            derivative (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class
        Returns:
            prev_sample (TODO): updated sample in the diffusion chain. derivative (TODO): TODO
        """
        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (sample_prev, derivative)

        return KarrasVeOutput(
            prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
        )

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()