# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import copy

from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.criterions.cross_entropy import CrossEntropyCriterion


from .ngram_s2s_model import NgramTransformerProphetModel
from .bert_dictionary import BertDictionary


@register_task('prophetnet_cosmo')
class ProphetnetMoETask(TranslationTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--num-experts', default=20, type=int, metavar='N',
                            help='number of experts')
        parser.add_argument('--gen-expert', type=int, default=0,
                            help='which expert to use for generation')
                            
        parser.add_argument('--device', action='store', type=str, 
                             default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        
    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def build_model(self, args):
        #from fairseq import models
        model = NgramTransformerProphetModel.build_model(args, self)
        return model

    def expert_index(self, i):
        return i + self.tgt_dict.index('<expert_0>')

    def _get_loss(self, sample, model, criterion):
        
        def extend_sample(samp):
            src_id = []
            src_tokens = []
            src_lengths = []
            target = []
            prev_output_tokens = []
            tokens = []
            max_l = 0
            for i in range(samp['target'].shape[0]): 
                splits = [0]
                for j in range(samp['target'][i].shape[0]):
                    if samp['target'][i][j] == 2: splits.append(j + 1)
                for j in range(len(splits) - 1):
                    src_tokens.append(samp['net_input']['src_tokens'][i].reshape(1,samp['net_input']['src_tokens'][i].shape[0]))
                    src_lengths.append(samp['net_input']['src_lengths'][i].view(1,1))
                    src_id.append(samp['id'][i].view(1,1))
                    if j == len(splits) - 1: 
                        t = samp['target'][i][splits[j]:splits[j + 1] - 1].view(1,splits[j+1] - splits[j] - 1)
                        t = F.pad(t, pad=(0, samp['target'][i].shape[0] - splits[j+1] + splits[j] + 1), mode='constant', value=0)
                        target.append(t)
                    else:
                        t = torch.cat((samp['target'][i][splits[j]:splits[j + 1] - 1].reshape(1,splits[j+1] - splits[j] - 1), \
                                       samp['target'][i][-1].reshape(1,1)), dim = 1)
                        t = F.pad(t, pad=(0, samp['target'][i].shape[0] - splits[j+1] + splits[j] ), mode='constant', value=0)
                        target.append(t)
                if len(splits) - 1 > max_l: max_l = len(splits) - 1
                
                splits = [0]
                for j in range(samp['net_input']['prev_output_tokens'][i].shape[0]):
                    if samp['net_input']['prev_output_tokens'][i][j] == 2: splits.append(j + 1)
                for j in range(len(splits) - 1):
                    if j == 0: 
                        t = samp['net_input']['prev_output_tokens'][i][splits[j]:splits[j + 1] - 1].view(1,splits[j+1] - splits[j] - 1)
                        t = F.pad(t, pad=(0, samp['net_input']['prev_output_tokens'][i].shape[0] - splits[j+1] + splits[j] + 1), mode='constant', value=0)
                        prev_output_tokens.append(t)
                    else:
                        t = torch.cat((samp['net_input']['prev_output_tokens'][i][0].reshape(1,1), \
                                       samp['net_input']['prev_output_tokens'][i][splits[j]:splits[j + 1] - 1].reshape(1,splits[j+1] - splits[j] - 1)), dim = 1)
                        t = F.pad(t, pad=(0, samp['net_input']['prev_output_tokens'][i].shape[0] - splits[j+1] + splits[j] ), mode='constant', value=0)
                        prev_output_tokens.append(t)
            
            target = torch.cat(target)
            prev_output_tokens = torch.cat(prev_output_tokens)
            src_tokens = torch.cat(src_tokens)
            src_lengths = torch.cat(src_lengths, dim = 1)[0]
            src_id = torch.cat(src_id, dim = 1)[0]
            nsentences = target.shape[0]
            ntokens = samp['ntokens']
            updated_sample = {}
            updated_sample['id'] = src_id
            updated_sample['target'] = target
            updated_sample['net_input'] = {}
            updated_sample['net_input']['src_tokens'] = src_tokens
            updated_sample['net_input']['src_lengths'] = src_lengths
            updated_sample['net_input']['prev_output_tokens'] = prev_output_tokens
            updated_sample['nsentences'] = nsentences
            updated_sample['ntokens'] = ntokens
            
            return updated_sample, max_l



        def modify_sample( sam, expert_idx, respected_z = None):
            #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            mod_sam = copy.deepcopy(sam)
            src_tokens_first = mod_sam['net_input']['src_tokens'][:,:1]
            src_tokens_second = mod_sam['net_input']['src_tokens'][:,1:]
            tmp = torch.LongTensor(mod_sam['net_input']['src_tokens'].shape[0],1).to(self.args.device)
            if not respected_z: tmp = tmp.fill_(self.expert_index(expert_idx))
            else:
                   for idx in range(len(respected_z)): tmp[idx] = self.expert_index(respected_z[idx])

            tmp = tmp.view(mod_sam['net_input']['src_tokens'].shape[0],1)
            mod_sam['net_input']['src_tokens'] = torch.cat([src_tokens_first,tmp, src_tokens_second], dim = 1)

            mod_sam['net_input']['src_lengths'] = mod_sam['net_input']['src_lengths'] + 1
            mod_sam['ntokens'] += 1
            
            mod_sam['id']
            mod_sam['net_input']['src_tokens']
            mod_sam['net_input']['src_lengths']
            mod_sam['net_input']['prev_output_tokens']
            mod_sam['target']
            
            return mod_sam

        
        def make_list(loss, sample_id, max_z):
            l_x = []
            l_y = []
            l_z = []
            split = []
            for idx in range(len(sample_id) - 1):
                if sample_id[idx + 1] != sample_id[idx]: split.append(idx+1)
                
            x = 1
            y = 1
            z = 0
            l = []
            for i in range(loss.shape[0]):
                if i % sample_id.shape[0] in split: 
                    x += 1 
                    y = 1
                if i % sample_id.shape[0] == 0:
                    x = 1
                    y = 1
                    z += 1
                l_x.append(x)
                l_y.append(y)
                l_z.append(z)
                y += 1
            return (l_x,l_y,l_z)
                
        def get_respected_z(indices, aux):
            tmp = {}
            final = []
            for i in indices:
                s = aux[0][i]
                d = aux[1][i]
                y = aux[2][i]
                if s not in tmp.keys():
                    tmp[s]= {}
                    tmp[s]['y'] = [d]
                    tmp[s]['z'] = [y]
                else:
                    if d in tmp[s]['y']: 
                        pass
                    elif y in tmp[s]['z']: 
                        pass
                    else:
                        tmp[s]['y'].append(d)
                        tmp[s]['z'].append(y)

            for k in tmp.keys():
                for idx in range(len(tmp[k]['y'])):
                    final.append(int(str(k)+ str(tmp[k]['y'][idx])+str(tmp[k]['z'][idx])))
                    #final.append(k * 100 + tmp[k]['y'][idx] * 10 + tmp[k]['z'][idx])
            final.sort()
            f = []
            for item in final: f.append(item%10)
            return f


        def get_lprob_yz( sam, max_z, winners=None):
                lprob_y = []
                for i in range(max_z):
                    tmp_sample = modify_sample(sam, i)
                    lprob_y.append(criterion.get_loss(model,tmp_sample))
                lprob_y = torch.cat(lprob_y)
                sorting_aux = make_list(lprob_y, sam['id'], max_z)
                _, sorted_indices = torch.sort(lprob_y)
                respected_z = get_respected_z(sorted_indices, sorting_aux)
                return respected_z


        sample, max_length = extend_sample(sample)
        # compute responsibilities without dropout
        with utils.eval(model):  # disable dropout
            with torch.no_grad():  # disable autograd
                respected_z = get_lprob_yz(sample, max_length)  # B x K
        
        sample = modify_sample(sample, 0, respected_z)
        loss = criterion(model,sample)[0]


        loss = loss.sum() 
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            #'posterior': prob_z_xy.float().sum(dim=0).cpu(),
        }
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, expert=None, prefix_tokens=None):

        def modify_sample( sam, expert_idx):
            #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            mod_sam = copy.deepcopy(sam)
            src_tokens_first = mod_sam['net_input']['src_tokens'][:,:1]
            src_tokens_second = mod_sam['net_input']['src_tokens'][:,1:]
            tmp = torch.LongTensor(mod_sam['net_input']['src_tokens'].shape[0],1).to(self.args.device)
            tmp = tmp.fill_(self.expert_index(expert_idx))

            tmp = tmp.view(mod_sam['net_input']['src_tokens'].shape[0],1)
            mod_sam['net_input']['src_tokens'] = torch.cat([src_tokens_first,tmp, src_tokens_second], dim = 1)

            mod_sam['net_input']['src_lengths'] = mod_sam['net_input']['src_lengths'] + 1

            return mod_sam
        expert = expert if expert else 0

        tmp_sample = modify_sample(sample, expert)
        with torch.no_grad():
            return generator.generate(
                models,
                tmp_sample,
                prefix_tokens=prefix_tokens,
                bos_token=None,
            )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        metrics.log_scalar(
            'posterior',
            sum(log['posterior'] for log in logging_outputs if 'posterior' in log)
        )

