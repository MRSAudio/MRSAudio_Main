
from utils.commons.hparams import hparams
from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0


def remove_slur(types, pitches, note_durs, ph_tokens):
    new_types = []
    new_pitches = []
    new_note_durs = []
    new_ph_tokens = []
    for i, t in enumerate(types):
        if t != 3:
            new_types.append(t)
            new_pitches.append(pitches[i])
            new_note_durs.append(note_durs[i])
            new_ph_tokens.append(ph_tokens[i])
    return new_types, new_pitches, new_note_durs, new_ph_tokens


class MIDIDataset(FastSpeechDataset):
    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(MIDIDataset, self).__getitem__(index)
        item = self._get_item(index)
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type
        
        # tech_len = len(item['ph'])
        # if 'mix_tech' not in item:
        #     mix = [2] * tech_len
        #     mix = torch.LongTensor(mix[:hparams['max_input_tokens']])
        # else:
        #     mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        
        # if 'falsetto_tech' not in item:
        #     falsetto = [2] * tech_len
        #     falsetto = torch.LongTensor(falsetto[:hparams['max_input_tokens']])
        # else:
        #     falsetto = torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        
        # if 'breathy_tech' not in item:
        #     breathe = [2] * tech_len
        #     breathe = torch.LongTensor(breathe[:hparams['max_input_tokens']])
        # else:
        #     breathe = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        
        # if 'bubble_tech' not in item:
        #     bubble = [2] * tech_len
        #     bubble = torch.LongTensor(bubble[:hparams['max_input_tokens']])
        # else:
        #     bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])
            
        # if 'strong_tech' not in item:
        #     strong = [2] * tech_len
        #     strong = torch.LongTensor(strong[:hparams['max_input_tokens']])
        # else:
        #     strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])
        
        # if 'weak_tech' not in item:
        #     weak = [2] * tech_len
        #     weak = torch.LongTensor(weak[:hparams['max_input_tokens']])
        # else:
        #     weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])

        # sample['mix'],sample['falsetto'],sample['breathe'],sample['bubble'],sample['strong'],sample['weak']=mix,falsetto,breathe,bubble,strong,weak

        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(MIDIDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        # mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        # falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        # breathe = collate_1d_or_2d([s['breathe'] for s in samples], 0.0)
        # bubble = collate_1d_or_2d([s['bubble'] for s in samples], 0.0)
        # strong = collate_1d_or_2d([s['strong'] for s in samples], 0.0)
        # weak = collate_1d_or_2d([s['weak'] for s in samples], 0.0)

        # batch['mix'],batch['falsetto'],batch['breathe'],batch['bubble'],batch['strong'],batch['weak']=mix,falsetto,breathe,bubble,strong,weak

        return batch


class MultiMIDIDataset(FastSpeechDataset):
    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(MultiMIDIDataset, self).__getitem__(index)
        item = self._get_item(index)
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type
        
        # tech_len = len(item['ph'])
        # if 'mix_tech' not in item:
        #     mix = [2] * tech_len
        #     mix = torch.LongTensor(mix[:hparams['max_input_tokens']])
        # else:
        #     mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        
        # if 'falsetto_tech' not in item:
        #     falsetto = [2] * tech_len
        #     falsetto = torch.LongTensor(falsetto[:hparams['max_input_tokens']])
        # else:
        #     falsetto = torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        
        # if 'breathy_tech' not in item:
        #     breathe = [2] * tech_len
        #     breathe = torch.LongTensor(breathe[:hparams['max_input_tokens']])
        # else:
        #     breathe = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        
        # if 'bubble_tech' not in item:
        #     bubble = [2] * tech_len
        #     bubble = torch.LongTensor(bubble[:hparams['max_input_tokens']])
        # else:
        #     bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])
            
        # if 'strong_tech' not in item:
        #     strong = [2] * tech_len
        #     strong = torch.LongTensor(strong[:hparams['max_input_tokens']])
        # else:
        #     strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])
        
        # if 'weak_tech' not in item:
        #     weak = [2] * tech_len
        #     weak = torch.LongTensor(weak[:hparams['max_input_tokens']])
        # else:
        #     weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])

        # if 'pharyngeal_tech' not in item:
        #     pharyngeal = [2] * tech_len
        #     pharyngeal = torch.LongTensor(pharyngeal[:hparams['max_input_tokens']])
        # else:
        #     pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])

        # if 'vibrato_tech' not in item:
        #     vibrato = [2] * tech_len
        #     vibrato = torch.LongTensor(vibrato[:hparams['max_input_tokens']])
        # else:
        #     vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])

        # if 'glissando_tech' not in item:
        #     glissando = [2] * tech_len
        #     glissando = torch.LongTensor(glissando[:hparams['max_input_tokens']])
        # else:
        #     glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])


        # sample['mix'],sample['falsetto'],sample['breathe'],sample['bubble'],sample['strong'],sample['weak']=mix,falsetto,breathe,bubble,strong,weak
        # sample['pharyngeal'],sample['vibrato'],sample['glissando']=pharyngeal,vibrato,glissando

        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(MultiMIDIDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        # mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        # falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        # breathe = collate_1d_or_2d([s['breathe'] for s in samples], 0.0)
        # bubble = collate_1d_or_2d([s['bubble'] for s in samples], 0.0)
        # strong = collate_1d_or_2d([s['strong'] for s in samples], 0.0)
        # weak = collate_1d_or_2d([s['weak'] for s in samples], 0.0)
        # pharyngeal = collate_1d_or_2d([s['pharyngeal'] for s in samples], 0.0)
        # vibrato = collate_1d_or_2d([s['vibrato'] for s in samples], 0.0)
        # glissando = collate_1d_or_2d([s['glissando'] for s in samples], 0.0)

        # batch['mix'],batch['falsetto'],batch['breathe'],batch['bubble'],batch['strong'],batch['weak']=mix,falsetto,breathe,bubble,strong,weak
        # batch['pharyngeal'],batch['vibrato'],batch['glissando']=pharyngeal,vibrato,glissando

        return batch