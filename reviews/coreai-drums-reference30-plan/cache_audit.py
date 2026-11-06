from pathlib import Path
import sys,json,hashlib
import torch
from transformers.cache_utils import DynamicCache,EncoderDecoderCache
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from export_coreai_dit_block import load_upstream,records,precision_metrics
p=root/'work/publisher';r=records()
for name in ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']:
 raw=(p/name).read_bytes();record=r[name]
 h=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() if record['algorithm']=='git-blob-sha1' else hashlib.sha256(raw).hexdigest()
 assert h==record['digest'],name
cfg,kind,rotary,mask=load_upstream(p/'modeling_acestep_v15_base.py',p/'config.json',p/'configuration_acestep_v15.py')
cfg.hidden_size=8;cfg.head_dim=4;cfg.num_attention_heads=2;cfg.num_key_value_heads=2;cfg.layer_types=['full_attention'];cfg.attention_dropout=0;cfg._attn_implementation='eager'
torch.manual_seed(901);attention=sys.modules[kind.__module__].AceStepAttention(cfg,0,is_cross_attention=True).eval()
encoder=torch.randn(2,7,8);queries=[torch.randn(2,9,8),torch.randn(2,9,8)];cache=EncoderDecoderCache(DynamicCache(),DynamicCache())
report={'scope':'Pinned publisher attention class with tiny synthetic random8dim weights; cache API/mechanism only, not learned full DiT or music validation','calls':4,'matched_query_pair_metrics':[]}
with torch.no_grad():
 for q in queries:
  cached=attention(hidden_states=q,attention_mask=None,encoder_hidden_states=encoder,past_key_value=cache)[0]
  plain=attention(hidden_states=q,attention_mask=None,encoder_hidden_states=encoder,past_key_value=None)[0]
  metric=precision_metrics(cached,plain);assert metric['parity_passed'];report['matched_query_pair_metrics'].append(metric)
report['self_attention_cache_sequence_length']=cache.get_seq_length();report['cross_attention_cached_length']=cache.cross_attention_cache.get_seq_length();report['cross_layer_updated']=cache.is_updated[0]
assert report['self_attention_cache_sequence_length']==0 and report['cross_attention_cached_length']==7
(root/'work/drum-reference30-v1/cache-mechanism-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
