from models.utils import (
    interpolate_pos_embed,
    interpolate_pos_relative_bias_beit,
    _init_transformer_weights,
    interpolate_temporal_pos_embed
)
from transformers import BeitModel, BeitConfig, ViTModel, ViTConfig
from models.xbert import BertConfig, BertModel, BertForMaskedLM, BertEncoder

import torch
from torch import nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SingularityRetrievalBase(nn.Module):
    """Common utils shared by pretraining and downstream retrieval"""
    def __init__(self, config=None, tokenizer=None, pretrain=True, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_dim = config.embed_dim
        self.vision_width = 768
        self.text_width = 768
        self.pretrain = pretrain

        self.vision_encoder, self.vision_layernorm = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config.temp)
        self.temp_negative = nn.Parameter(torch.ones([]) * config.temp_neg_ratio)
        self.itm_head = nn.Linear(self.text_width, 2)

        # video temporal embedding,
        # TODO: should we just interpolate instead of initializing more temporal embed?
        # https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L278
        self.temporal_embeddings = nn.Parameter(
            torch.zeros(1, config.video_input.num_frames, 1, self.vision_width))
        if config.temporal_vision_encoder.enable:
            self.temporal_vision_encoder = self.build_temporal_vision_encoder()

    def forward(self, image, text, idx, neg_text1=None, neg_text2=None, neg_text3=None, neg_text4=None, neg_text5=None, neg_text6=None, neg_text7=None):
        # ================= Dual Encoder ITC loss ================ #
        self.clip_contrastive_temperature()
        self.clip_contrastive_temperature_negative()
        
        image_embeds, pooled_image_embeds = self.encode_image(image)
        text_embeds, pooled_text_embeds = self.encode_text(text)
        neg_text1_embeds, pooled_text1_embeds = self.encode_text(neg_text1)
        neg_text2_embeds, pooled_text2_embeds = self.encode_text(neg_text2)
        neg_text3_embeds, pooled_text3_embeds = self.encode_text(neg_text3)
        neg_text4_embeds, pooled_text4_embeds = self.encode_text(neg_text4)
        neg_text5_embeds, pooled_text5_embeds = self.encode_text(neg_text5)
        neg_text6_embeds, pooled_text6_embeds = self.encode_text(neg_text6)
        neg_text7_embeds, pooled_text7_embeds = self.encode_text(neg_text7)
        print('text in forward', text_embeds.shape)
        print('pooled_text_embeds in forward', pooled_text_embeds.shape)
        print('image in forward', image_embeds.shape)
        print('pooled_image_embeds in forward', pooled_image_embeds.shape)
        print('neg_text1_embeds in forward', neg_text1_embeds.shape)

        loss_ita, sim_i2t, sim_t2i = self.get_contrastive_loss(
            pooled_image_embeds, pooled_text_embeds, idx, pooled_text1_embeds, pooled_text2_embeds, pooled_text3_embeds, pooled_text4_embeds, pooled_text5_embeds, pooled_text6_embeds, pooled_text7_embeds)

        # ================= Multi-Modal Encoder ITM loss ================ #
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        loss_itm, ce_cls_embeds_pos = self.get_itm_loss(
            sim_i2t, sim_t2i, text_embeds, text.attention_mask,
            image_embeds, image_atts, idx=idx)

        return_dict = dict(
            loss_ita=loss_ita * self.config.loss_weight.itc,
            loss_itm=loss_itm * self.config.loss_weight.itm
        )

        # ================= Multi-Modal Encoder MLM loss ======================== #
        if self.pretrain:
            loss_mlm = self.get_mlm_loss(text, image_embeds, image_atts)
            return_dict.update(loss_mlm=loss_mlm * self.config.loss_weight.mlm)

        return return_dict

    def build_text_encoder(self):
        logger.info(f"Build text_encoder {self.config.text_encoder}")
        bert_config = BertConfig.from_json_file(self.config.bert_config)
        if self.pretrain:
            text_encoder, loading_info = BertForMaskedLM.from_pretrained(
                self.config.text_encoder, config=bert_config,
                output_loading_info=True
            )
        else:
            text_encoder, loading_info = BertModel.from_pretrained(
                self.config.text_encoder, config=bert_config,
                add_pooling_layer=False, output_loading_info=True
            )
        if self.config.debug:
            for k, v in loading_info.items():
                logger.debug(f"loading_info[{k}]: {v}")
        logger.info(f"Build text_encoder {self.config.text_encoder}, done!")
        return text_encoder

    def build_vision_encoder(self):
        if self.config.vit_type in ["beit", "deit", "vit"]:
            vision_encoder = self.build_huggingface_vit_with_image_size(
                self.config.vit_name_or_pretrained_path, self.config.image_res)
        else:
            raise ValueError(f"Unknown vit type {self.config.vit_type}")

        # add layernorm for normalizing BEiT outputs hidden states
        vision_layernorm = None
        if self.config.vit_type == "beit":
            vision_layernorm = nn.LayerNorm(self.vision_width, eps=1e-12)
        return vision_encoder, vision_layernorm

    def build_temporal_vision_encoder(self):
        n_layer = self.config.temporal_vision_encoder.num_layers
        logger.info(f"Build temporal_vision_encoder (#layer={n_layer}), randomly initialised.")
        bert_config = BertConfig.from_json_file(self.config.bert_config)
        bert_config.num_hidden_layers = n_layer
        bert_config.fusion_layer = n_layer

        temporal_vision_encoder = BertEncoder(bert_config)
        _init_transformer_weights(temporal_vision_encoder, bert_config.initializer_range)
        logger.info(f"Build temporal_vision_encoder, done!")
        return temporal_vision_encoder

    @classmethod
    def build_huggingface_vit_with_image_size(cls, model_card: str, image_size: int):
        """Build a vit model from huggingface hub, also interpolate pos_embed when needed.

        Args:
            model_card: name in huggingface hub, e.g., `facebook/deit-base-patch16-224`
            image_size: new image size, may be different from pre-training image_size of `model_card`

        ref: https://github.com/huggingface/transformers/issues/12167#issuecomment-861356232
        """
        is_beit = "beit" in model_card
        if "beit" in model_card:
            model_cls, config_cls = BeitModel, BeitConfig
        elif "deit" in model_card or "vit" in model_card:
            # the deit model we use is loaded in vit arch,
            # see https://huggingface.co/facebook/deit-base-patch16-224#how-to-use
            model_cls, config_cls = ViTModel, ViTConfig
        else:
            raise ValueError(f"Unexpected model_card: {model_card}")

        logger.info(f"Loading vit pre-trained weights from huggingface {model_card}.")
        # BEiT uses average pooled tokens instead of [CLS] used by other models
        tmp_model = model_cls.from_pretrained(model_card, add_pooling_layer=is_beit)
        state_dict = tmp_model.state_dict()
        del tmp_model

        logger.info(f"Init new model with new image size {image_size}, and load weights.")
        model_config = config_cls.from_pretrained(model_card, image_size=image_size)
        model = model_cls(config=model_config, add_pooling_layer=is_beit)
        if is_beit:
            # interpolate relative pos bias
            state_dict = interpolate_pos_relative_bias_beit(
                state_dict_old=state_dict,
                state_dict_new=model.state_dict(),
                patch_shape_new=model.embeddings.patch_embeddings.patch_shape
            )
        else:
            # interpolate pos_embed and load weights to new model
            state_dict["embeddings.position_embeddings"] = interpolate_pos_embed(
                pos_embed_old=state_dict["embeddings.position_embeddings"],
                pos_embed_new=model.embeddings.position_embeddings,
                num_patches_new=model.embeddings.patch_embeddings.num_patches
            )
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        return model

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def encode_image(self, image):
        bsz, num_frms, c, h, w = image.shape  # `num_frms` could be changing for image (=1) or video (e.g., =4)
        if self.config.add_temporal_embed:
            num_temp = self.temporal_embeddings.shape[1]
            if num_temp >= num_frms:
                temporal_embeddings = self.temporal_embeddings[:, :num_frms]  # might be redefined
            else:  # interpolate or add zero
                temporal_embeddings = interpolate_temporal_pos_embed(
                    self.temporal_embeddings, num_frms)
        else:
            temporal_embeddings = 0.  # equivalent to no temporal embedding

        image_embeds, pooled_image_embeds = self._encode_image(image)

        image_embeds = image_embeds + temporal_embeddings  # (bsz, num_frms, L, d)
        if num_frms > 1 and self.config.temporal_vision_encoder.enable:  # skip if single frame or disabled.
            image_embeds, _pooled_image_embeds = self.temporal_encode_frames(image_embeds)  # (bsz, num_frms, L, d)
            if self.config.temporal_vision_encoder.update_pooler_embed:
                pooled_image_embeds = _pooled_image_embeds

        image_embeds = image_embeds.view(bsz, -1, self.vision_width)  # (bsz, num_frms*L, d)
        return image_embeds, pooled_image_embeds  # (bsz, #frm*L, d), (bsz, #frm, d)

    def temporal_encode_frames(self, image_embeds):
        # perform temporal self-attention, each token sees all other tokens at same spatial location across frames
        bsz, num_frms, spatial_seq_l, hsz = image_embeds.shape
        image_embeds = image_embeds.transpose(1, 2).reshape(bsz*spatial_seq_l, num_frms, -1)  # (bsz*L, num_frms, d)
        image_embeds = self.temporal_vision_encoder(image_embeds, mode="temporal")   # dict
        image_embeds = image_embeds.last_hidden_state  # (bsz*L, num_frms, d)
        image_embeds = image_embeds.reshape(bsz, spatial_seq_l, num_frms, -1)
        image_embeds = image_embeds.transpose(1, 2).contiguous()  # (bsz, num_frms, L, d)
        # take [CLS] embeds, maybe mean-pooling (image_embeds[:, :, 1:].mean(2)) as in BEiT is better?
        pooled_image_embeds = image_embeds[:, :, 0]  # (bsz, num_frms, d)
        return image_embeds, pooled_image_embeds

    def _encode_image(self, image):
        bsz, num_frms, c, h, w = image.shape  # `num_frms` could be changing for image (=1) or video (e.g., =4)
        image = image.view(bsz*num_frms, c, h, w)
        image_embeds = self.vision_encoder(image)
        if self.vision_layernorm is not None:  # only for BEiT mean-pooling
            image_embeds.last_hidden_state = self.vision_layernorm(image_embeds.last_hidden_state)

        if self.config.vit_type == "beit":
            pooled_image_embeds = image_embeds.pooler_output  # (bsz*num_frms, d)
            image_embeds = image_embeds.last_hidden_state  # (bsz*num_frms, L, d)
        else:
            image_embeds = image_embeds.last_hidden_state
            pooled_image_embeds = image_embeds[:, 0]

        image_embeds = image_embeds.view(bsz, num_frms, -1, self.vision_width)  # (bsz, num_frms, L, d)
        pooled_image_embeds = pooled_image_embeds.view(bsz, num_frms, self.vision_width) \
            if pooled_image_embeds is not None else None  # (bsz, num_frms, d)
        return image_embeds, pooled_image_embeds

    def encode_text(self, text):
        # text
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text'
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        #self.temp.clamp_(min_val, max_val)
        
    def clip_contrastive_temperature_negative(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        #self.temp.clamp_(min_val, max_val)

    @torch.no_grad()
    def get_mask(self, sim, idx=None, normalize=False):
        """
        sim: (N, N)
        idx: (N, )
        normalize: bool, make row sum equal to 1
        """
        if idx is not None:
            idx = idx.view(-1, 1)
            mask = torch.eq(idx, idx.T).to(sim.dtype)
            if normalize:  
                mask = mask / mask.sum(1, keepdim=True)
        else:
            mask = torch.zeros_like(sim)
            mask.fill_diagonal_(1)
        return mask  # `1` mark valid/matched location

    def get_contrastive_loss(self, pooled_image_embeds, pooled_text_embeds, idx=None, neg_text1_embeds=None, neg_text2_embeds=None, neg_text3_embeds=None, neg_text4_embeds=None, neg_text5_embeds=None, neg_text6_embeds=None, neg_text7_embeds=None):
        sim_i2t, sim_t2i = self.get_sim(
            pooled_image_embeds, pooled_text_embeds, t=self.temp)
        
        with torch.no_grad():
            sim_i2t_targets = self.get_mask(sim_i2t, idx=idx, normalize=True)
            #send sim_i2t_targets to gpu
            sim_i2t_targets = sim_i2t_targets.to(pooled_image_embeds.device)
            #create a new tensor with same size as sim_i2t_targets and only diagonal value is 1
            sim_i2t_targets_inbatch = torch.eye(sim_i2t_targets.size(0))
            #send sim_i2t_targets to gpu
            sim_i2t_targets_inbatch = sim_i2t_targets_inbatch.to(neg_text1_embeds.device)
            
            sim_t2i_targets = sim_i2t_targets
            #send sim_t2i_targets to gpu
            sim_t2i_targets = sim_t2i_targets.to(pooled_image_embeds.device)
            

        image_proj = self.vision_proj
        text_proj = self.text_proj
        #init all neg_text_embeds to None
        neg_text1_embeds_inbatch = None
        neg_text2_embeds_inbatch = None
        neg_text3_embeds_inbatch = None
        neg_text4_embeds_inbatch = None
        neg_text5_embeds_inbatch = None
        neg_text6_embeds_inbatch = None
        neg_text7_embeds_inbatch = None
        pos_text_feat = F.normalize(text_proj(pooled_text_embeds), dim=-1)
        image_feat = F.normalize(image_proj(pooled_image_embeds), dim=-1)
        
        bs = len(pooled_image_embeds)
        if self.config.evaluate is False:
            
            for i in range(len(pooled_image_embeds)):    #len:bsz
                neg_text1_embeds_inbatch = F.normalize(text_proj(neg_text1_embeds[i]), dim=-1)
                neg_text2_embeds_inbatch = F.normalize(text_proj(neg_text2_embeds[i]), dim=-1)
                neg_text3_embeds_inbatch = F.normalize(text_proj(neg_text3_embeds[i]), dim=-1)
                neg_text4_embeds_inbatch = F.normalize(text_proj(neg_text4_embeds[i]), dim=-1)
                neg_text5_embeds_inbatch = F.normalize(text_proj(neg_text5_embeds[i]), dim=-1)
                neg_text6_embeds_inbatch = F.normalize(text_proj(neg_text6_embeds[i]), dim=-1)
                neg_text7_embeds_inbatch = F.normalize(text_proj(neg_text7_embeds[i]), dim=-1)
           
                pos_image_feat = image_feat[i]
                #pos_image_feat size 4 * 256
                #make it 1 * 4 * 256
                
                pos_image_feat = pos_image_feat.unsqueeze(0)
                #print('pos_iamge_feat in get_contrastive_loss', pos_image_feat.shape)
                #concate all neg_text_embeds
                #neg_text1_embeds_inbatch : 256
                #unsqueeze to 1 * 256
                #caoncate all neg_text_embeds size 7 * 256
                neg_text_embeds_inbatch = torch.cat((neg_text1_embeds_inbatch.unsqueeze(0), neg_text2_embeds_inbatch.unsqueeze(0), neg_text3_embeds_inbatch.unsqueeze(0), neg_text4_embeds_inbatch.unsqueeze(0), neg_text5_embeds_inbatch.unsqueeze(0), neg_text6_embeds_inbatch.unsqueeze(0), neg_text7_embeds_inbatch.unsqueeze(0)), 0)
                
                sim_i2t_pos = torch.einsum("mld,nd->mln", pos_image_feat, pos_text_feat).mean(1) / self.temp
                #print('sim_i2t_pos in get_contrastive_loss', sim_i2t_pos.shape)
                if self.config.temp_neg:
                    print('temp_neg is True')
                    sim_i2t_neg = torch.einsum("mld,nd->mln", pos_image_feat, neg_text_embeds_inbatch).mean(1) / self.temp
                elif self.config.temp_neg is False:
                    print('temp_neg is False')
                    sim_i2t_neg = torch.einsum("mld,nd->mln", pos_image_feat, neg_text_embeds_inbatch).mean(1) / self.temp_negative
                #print('sim_i2t_neg in get_contrastive_loss', sim_i2t_neg.shape)
                #concate pos and neg sim_i2t
                sim_i2t_extended = torch.cat((sim_i2t_pos, sim_i2t_neg), 1)
                #print('sim_i2t_extended in get_contrastive_loss', sim_i2t_extended.shape)
                if i == 0:
                    extended_sim_matrix  = sim_i2t_extended   
                else:
                    extended_sim_matrix = torch.cat((extended_sim_matrix, sim_i2t_extended), 0)
                
            #print('extended_sim_matrix in get_contrastive_loss', extended_sim_matrix)
            
            
            
            #get max value matrix of each row in extended_sim_matrix
            #max_value = torch.max(extended_sim_matrix,dim=1)[0]
            
            #print('max_value in get_contrastive_loss', max_value)
            #print('max_value in get_contrastive_loss', max_value.shape)
            #avoid nan
            #extended_sim_matrix = extended_sim_matrix - max_value.unsqueeze(1)
            #print('extended_sim_matrix in get_contrastive_loss', extended_sim_matrix)
            #print('extended_sim_matrix in get_contrastive_loss', extended_sim_matrix)
            softmax_vt = torch.exp(extended_sim_matrix)
            #print('softmax_vt in get_contrastive_loss', softmax_vt)
            
            part_1 = torch.sum(softmax_vt[:, 0:bs], dim=1)
            part_2 = torch.sum(softmax_vt[:, bs+1:], dim=1)
            
            
            sim_i2t_inbatch = torch.einsum("mld,nd->mln", image_feat, pos_text_feat).mean(1) / self.temp
            
            # sim_i2t_inbatch minus max_value
            #sim_i2t_inbatch = sim_i2t_inbatch - max_value.unsqueeze(1)
            #print('sim_i2t_inbatch in get_contrastive_loss', sim_i2t_inbatch)
            loss_i2t_orig = -torch.sum(
                F.log_softmax(sim_i2t_inbatch, dim=1) * sim_i2t_targets_inbatch, dim=1).mean()
            
            sim_i2t_inbatch = torch.exp(sim_i2t_inbatch)
            
            
            #get loss
            gt = torch.diag(sim_i2t_inbatch)
            loss_i2t_exp = -torch.log(gt/(part_1 + self.config.neg_ratio * part_2))
            print('neg_ratio in get_contrastive_loss', self.config.neg_ratio)
            loss_i2t = torch.mean(loss_i2t_exp)
            
            loss_t2i = -torch.sum(
                F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            
            loss_ita = (loss_i2t + loss_t2i) / 2
            
            
            
            
            print('loss_i2t_orig in get_contrastive_loss', loss_i2t_orig)
            print('loss_i2t in get_contrastive_loss', loss_i2t)
        
        
        
        if self.config.evaluate:
            loss_i2t = -torch.sum(
                F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(
                F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            
            loss_v2t_with_neg = 0

            loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita, sim_i2t, sim_t2i

    def get_sim(self, pooled_image_embeds, pooled_text_embeds, t=1):
        """
        Args:
            pooled_image_embeds: (bsz, num_frms, d)
            pooled_text_embeds: (bsz, d)
            t: temperature
        """
        image_proj = self.vision_proj
        text_proj = self.text_proj

        image_feat = F.normalize(image_proj(pooled_image_embeds), dim=-1)
        text_feat = F.normalize(text_proj(pooled_text_embeds), dim=-1)

        #save image and text features only when evaluating
        if self.config.evaluate:
            manipulation = ''
            
            if image_feat.size(0) == 184:
                manipulation = 'temporal_contact_swap_mani'
            elif image_feat.size(0) == 62:
                manipulation = 'temporal_action_swap_mani'
            elif image_feat.size(0) == 102:
                manipulation = 'neighborhood_same_entity_mani'
            elif image_feat.size(0) == 35:
                manipulation = 'neighborhood_different_entity_mani'
            elif image_feat.size(0) == 935:
                manipulation = 'counter_spatial_mani'
            elif image_feat.size(0) == 1008:
                manipulation = 'counter_contact_mani'
            elif image_feat.size(0) == 1168:
                manipulation = 'counter_action_mani'
            elif image_feat.size(0) == 818:
                manipulation = 'counter_attribute_mani'
            else:
                #throw an exception
                raise Exception('Unknown manipulation type')
            
            # if image_feat.size(0) == 563:
            #     manipulation = 'temporal_int_mani'
            # elif image_feat.size(0) == 489:
            #     manipulation = 'temporal_act_mani'
            # elif image_feat.size(0) == 160:
            #     manipulation = 'neighborhood_same_entity_mani'
            # elif image_feat.size(0) == 597:
            #     manipulation = 'neighborhood_diff_entity_mani'
            # elif image_feat.size(0) == 583:
            #     manipulation = 'counter_rel_mani'
            # elif image_feat.size(0) == 591:
            #     manipulation = 'counter_int_mani'
            # # elif image_feat.size(0) == 591:
            # #     manipulation = 'counter_attr_mani'
            # elif image_feat.size(0) == 535:
            #     manipulation = 'counter_act_mani'
            # else:
                # #throw an exception
                # raise Exception('Unknown manipulation type')
            
            print("saving image and text features of type: ", manipulation)
            print(text_feat.size())
            print(image_feat.size())
            text_feat_save_path = self.config.output_dir + '/text_feat_' + manipulation + '.pt'
            image_feat_save_path = self.config.output_dir + '/image_feat_' + manipulation + '.pt'
            torch.save(text_feat, text_feat_save_path)
            torch.save(image_feat, image_feat_save_path)
            # check if the features are saved
            import os
            print(os.path.exists(text_feat_save_path))
            print(os.path.exists(image_feat_save_path))
            print("saved image and text features of type: ", manipulation)                 
        #     
        sim_i2t = torch.einsum("mld,nd->mln", image_feat, text_feat).mean(1) / t  # (N, N)
        
        sim_t2i = sim_i2t.T
        return sim_i2t, sim_t2i

    def get_itm_loss(
            self, sim_i2t, sim_t2i, text_embeds, text_atts, image_embeds, image_atts, idx=None):
        """
        sim_i2t, sim_t2i: (N, N)
        text_embeds, text_atts, image_embeds, image_atts: (N, *)
        idx: (N, )
        """
        bsz = len(sim_i2t)
        text_encoder = self.get_text_encoder()

        with torch.no_grad():
            weights_i2t = F.softmax(sim_i2t+1e-4, dim=1)  # (N, N)
            weights_t2i = F.softmax(sim_t2i+1e-4, dim=1)

            mask = self.get_mask(sim_i2t, idx=idx).bool()
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

        # select a negative image for each text
        if self.config.itm_hard_neg:
            img_neg_indices = torch.multinomial(weights_t2i, 1).squeeze()
        else:
            img_neg_indices = self.get_rand_indices(mask, 1).squeeze()            

        image_embeds_neg = image_embeds[img_neg_indices]

        # select a negative text for each image
        if self.config.itm_hard_neg:
            txt_neg_indices = torch.multinomial(weights_i2t, 1).squeeze()
        else:
            txt_neg_indices = self.get_rand_indices(mask, 1).squeeze()

        text_embeds_neg = text_embeds[txt_neg_indices]
        text_atts_neg = text_atts[txt_neg_indices]  # (N, L, d)

        # embedding on local gpu
        _text_embeds = text_embeds  
        _text_atts = text_atts
        _image_embeds = image_embeds
        _image_atts = image_atts
        # concat embeddings
        text_embeds_all = torch.cat([_text_embeds, _text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([_text_atts, _text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([_image_embeds, image_embeds_neg, _image_embeds], dim=0)
        image_atts_all = torch.cat([_image_atts, _image_atts, _image_atts], dim=0)
        output = text_encoder(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode='fusion',
        )

        itm_embeds = output.last_hidden_state[:, 0]  # pos (N, d) + neg (2N, d)

        loss_itm = self._get_itm_loss(itm_embeds, enc=self.itm_head)
        itm_embeds_pos = itm_embeds[:bsz]  # (N, d)

        return loss_itm, itm_embeds_pos

    def _get_itm_loss(self, itm_embeds, enc):
        """
        itm_embeds: (3*N, D)
        enc: nn.Module that projects cls_embeds
        """
        itm_scores = enc(itm_embeds)  # (3*N, 2)
        bs = itm_scores.size(0) // 3
        itm_labels = itm_scores.new_ones(3*bs, dtype=torch.long)
        itm_labels[bs:] = 0
        loss_itm = F.cross_entropy(itm_scores, itm_labels)
        return loss_itm

    def get_rand_indices(self, mask, k):
        """
        Args:
            mask: (N, L) 0 indicates the positions that we can sample, 1 otherwise
            k: #indices to sample at each row
        Returns:
            (N, k) indices
        """
        mask = mask.float()
        mask = mask - 10000 * mask
        mask += torch.randn_like(mask)
        _, indices = torch.sort(mask, dim=1, descending=True)
        indices = indices[:, :k].contiguous()  
        return indices
