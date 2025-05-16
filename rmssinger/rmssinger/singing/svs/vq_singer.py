from singing.svs.module.vq_singer import VQSinger, DiffPostnet, MultiVQSinger
from singing.svs.base_gen_task import AuxDecoderMIDITask
from utils.commons.hparams import hparams
import torch
import torch.nn.functional as F
from modules.tts.iclspeech.multi_window_disc import Discriminator
from singing.svs.dataset import MIDIDataset, MultiMIDIDataset
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
import matplotlib.pyplot as plt

def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='ref')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig

class VQSingerTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = MIDIDataset
        self.mse_loss_fn = torch.nn.MSELoss()

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = VQSinger(dict_size, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def drop_multi(self, tech, drop_p):
        random_tech = torch.rand_like(tech, dtype=torch.float32)
        tech[random_tech < drop_p] = 2
        return tech
    
    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        ph_lengths = sample['txt_lengths']
        if hparams['use_spk_id']:
            spk_id = sample["spk_ids"]
        else:
            spk_id=None
        if hparams['use_spk_embed']==True:
            spk_embed=sample['spk_embed']
        else:
            spk_embed=None
            
        # if hparams.get("wotech", False):
        #     tech_drop = {
        #         'mix': 1,
        #         'falsetto': 1,
        #         'breathe': 1,
        #         'bubble': 1,
        #         'strong': 1,
        #         'weak': 1
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        # if not infer and hparams.get("tech_drop", False):
        #     tech_drop = {
        #         'mix': 0.1,
        #         'falsetto': 0.1,
        #         'breathe': 0.1,
        #         'bubble': 0.1,
        #         'strong': 0.1,
        #         'weak': 0.1
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
                
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        # mix,falsetto,breathe,bubble,strong,weak=sample['mix'],sample['falsetto'],sample['breathe'],sample['bubble'],sample['strong'],sample['weak']
        target = sample["mels"]  
        # print(spk_id)
        output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                            target=target,ph_lengths=ph_lengths, f0=f0, uv=uv, infer=infer, 
                            # mix=mix,falsetto=falsetto,breathe=breathe,bubble=bubble,strong=strong,weak=weak,
                            note=notes, note_dur=note_durs, note_type=note_types)
        losses = {}
        
        losses["diff"] = output["diff"]
        
        if not infer:
            self.add_mel_loss(output['mel_out'], target, losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)

        # if 'vq_loss' in output:
        #     losses['vq_loss']=output['vq_loss']
        # else:
        #     nonpadding = (txt_tokens > 0).float()
        #     vq_ph_loss = F.mse_loss(output['z_q_x'][0], output['z_e_x'][0].detach(), reduction='none')
        #     vq_ph_loss = (vq_ph_loss.mean(-2) * nonpadding).sum() / nonpadding.sum()
        #     losses['vq_ph'] = vq_ph_loss
        #     commit_ph_loss = F.mse_loss(output['z_e_x'][0], output['z_q_x'][0].detach(), reduction='none')
        #     commit_ph_loss = (commit_ph_loss.mean(-2) * nonpadding).sum() / nonpadding.sum()
        #     losses['commit_ph'] = commit_ph_loss

        return losses, output

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]
    
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {"diff": 0}
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            outputs['losses'], model_out = self.run_model(sample, infer=True)
            outputs['total_loss'] = sum(outputs['losses'].values())
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            wav_gt = self.vocoder.spec2wav(sample["mels"][0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy(), f0=model_out["f0_denorm_pred"][0].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], None, model_out["f0_denorm_pred"][0]),
                self.global_step)
        return outputs

class MultiVQSingerTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = MultiMIDIDataset
        self.mse_loss_fn = torch.nn.MSELoss()

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = MultiVQSinger(dict_size, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def drop_multi(self, tech, drop_p):
        random_tech = torch.rand_like(tech, dtype=torch.float32)
        tech[random_tech < drop_p] = 2
        return tech
    
    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        ph_lengths = sample['txt_lengths']
        if hparams['use_spk_id']:
            spk_id = sample["spk_ids"]
        else:
            spk_id=None
        if hparams['use_spk_embed']==True:
            spk_embed=sample['spk_embed']
        else:
            spk_embed=None
        
        # if hparams.get("wotech", False):
        #     tech_drop = {
        #         'mix': 1,
        #         'falsetto': 1,
        #         'breathe': 1,
        #         'bubble': 1,
        #         'strong': 1,
        #         'weak': 1,
        #         'pharyngeal': 1,
        #         'vibrato': 1,
        #         'glissando': 1,
                
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        # if not infer and hparams.get("tech_drop", False):
        #     tech_drop = {
        #         'mix': 0.1,
        #         'falsetto': 0.1,
        #         'breathe': 0.1,
        #         'bubble': 0.1,
        #         'strong': 0.1,
        #         'weak': 0.1,
        #         'pharyngeal': 0.1,
        #         'vibrato': 0.1,
        #         # 'glissando': 0.1,
        #     }
        #     for tech, drop_p in tech_drop.items():
                # sample[tech] = self.drop_multi(sample[tech], drop_p)
                
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        mix,falsetto,breathe,bubble,strong,weak=sample['mix'],sample['falsetto'],sample['breathe'],sample['bubble'],sample['strong'],sample['weak']
        pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
        target = sample["mels"]  
        # print(spk_id)
        output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                            target=target,ph_lengths=ph_lengths, f0=f0, uv=uv, infer=infer, 
                            # mix=mix,falsetto=falsetto,breathe=breathe,bubble=bubble,strong=strong,weak=weak,pharyngeal=pharyngeal,vibrato=vibrato,glissando=glissando,
                            note=notes, note_dur=note_durs, note_type=note_types)
        losses = {}
        
        losses["diff"] = output["diff"]
        
        if not infer:
            self.add_mel_loss(output['mel_out'], target, losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)

        # if 'vq_loss' in output:
        #     losses['vq_loss']=output['vq_loss']
        # else:
        #     nonpadding = (txt_tokens > 0).float()
        #     vq_ph_loss = F.mse_loss(output['z_q_x'][0], output['z_e_x'][0].detach(), reduction='none')
        #     vq_ph_loss = (vq_ph_loss.mean(-2) * nonpadding).sum() / nonpadding.sum()
        #     losses['vq_ph'] = vq_ph_loss
        #     commit_ph_loss = F.mse_loss(output['z_e_x'][0], output['z_q_x'][0].detach(), reduction='none')
        #     commit_ph_loss = (commit_ph_loss.mean(-2) * nonpadding).sum() / nonpadding.sum()
        #     losses['commit_ph'] = commit_ph_loss

        return losses, output

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]
    
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {"diff": 0}
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            outputs['losses'], model_out = self.run_model(sample, infer=True)
            outputs['total_loss'] = sum(outputs['losses'].values())
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            wav_gt = self.vocoder.spec2wav(sample["mels"][0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy(), f0=model_out["f0_denorm_pred"][0].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], None, model_out["f0_denorm_pred"][0]),
                self.global_step)
        return outputs
    
class DiffPostnetTask(VQSingerTask):
    def __init__(self):
        super(DiffPostnetTask, self).__init__()

    def build_model(self):
        self.build_pretrain_model()
        self.model = DiffPostnet()

    def build_pretrain_model(self):
        dict_size = len(self.token_encoder)
        self.pretrain = VQSinger(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    

    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        # spk_id = sample["spk_ids"]
        if hparams['use_spk_embed']==True:
            spk_embed=sample['spk_embed']
        else:
            spk_embed=None
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        ph_lengths = sample['txt_lengths']
        output = self.pretrain(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=None,target=target,ph_lengths=ph_lengths, f0=f0, uv=uv, infer=infer, note=notes, note_dur=note_durs, note_type=note_types)
        self.model(target, infer, output, spk_embed)
        losses = {}
        losses["diff"] = output["diff"]
        return losses, output
    
    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output
    
    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])
