"""CPU-only algebra for same-shape persistent main fields; not GPU measurements."""
import json
from pathlib import Path

N, d, m, L, H, k = 31668, 5120, 13824, 40, 40, 2560
GIB = 2**30
ours_ffn = 2*N*L*(2*d+k)
chip_ffn = 2*N*L*(m+d)
icc_ffn = 4*N*L*(m+d)
chip_attention_output = 2*N*L*d
ours_attention_main = chip_attention_output + 2*N*L*H*31 + 4*N*L*H
out = {
    'assumptions': {'N':N,'d':d,'m':m,'L':L,'H':H,'k':k,'BF16_bytes':2,'rho_bytes':4,'expert_scope':'one active expert','residency':'all L layers resident'},
    'FFN_main_fields_GiB': {'FFNDS':ours_ffn/GIB,'Chipmunk':chip_ffn/GIB,'ICC':icc_ffn/GIB},
    'FFNDS_FFN_reduction_fraction_vs': {'Chipmunk':1-ours_ffn/chip_ffn,'ICC':1-ours_ffn/icc_ffn},
    'attention_main_fields_GiB': {'Chipmunk_output_only':chip_attention_output/GIB,'FFNDS_history_feature_rho':ours_attention_main/GIB},
    'FFNDS_combined_main_fields_GiB':(ours_ffn+ours_attention_main)/GIB,
    'limitations':['not a theoretical storage lower bound','not peak GPU memory','same-shape source-layout comparison, not author Wan benchmark values','Chipmunk video benchmark disables MLP','attention masks/indices/counts/LSE/padding and other metadata are excluded from these MAIN-field sums and must be counted for total state','offload moves host state and changes HBM footprint; compare transfer cost too']
}
Path(__file__).with_name('STATE_CALCULATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
