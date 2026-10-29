"""Wan FFN author-SharQ physical call adapter; no GPU import on module load.

One linear emits sparse FP4 and dense FP4 residual GEMMs. The latter is a
full K-by-N product, not a compressed or skipped-K residual. Both share the
group32 quantized W payload and have different scale layouts. This module
provides no authorization, assets, scheduler or experiment entry point.
"""
ARMS=('original_bf16','dense_group16','dense_shared32','sharq')


def sharq_linear(x, layer, api, ledger):
    prepared=ledger.call('sharq_prepare',api.sharq_prepare,x,layer.out_features)
    a_comp, metadata, scale_sparse, a_residual, scale_residual, a_global=prepared
    output_scale=ledger.call('output_scale',api.output_scale,a_global,layer.weight_scale)
    sparse=ledger.call('sparse_gemm',api.sparse_gemm,a_comp,layer.weight_q,metadata,
                       scale_sparse,layer.scale_w_sparse,x.shape[0],layer.out_features,
                       layer.in_features,output_scale)
    result=ledger.call('dense_residual_gemm',api.dense_accum,a_residual,layer.weight_q,
                       scale_residual,layer.scale_w_dense,output_scale,sparse)
    return ledger.call('bias_add',api.bias_add,result,layer.bias)


def dense32_linear(x, layer, api, ledger):
    a,scale_a,a_global=ledger.call('dense32_prepare',api.dense_prepare,x)
    alpha=ledger.call('output_scale',api.output_scale,a_global,layer.weight_scale)
    result=ledger.call('dense32_gemm',api.dense_gemm,a,layer.weight_q,scale_a,
                       layer.scale_w_dense,alpha)
    return ledger.call('bias_add',api.bias_add,result,layer.bias)


def forward(arm, x, first, second, gate, api, ledger):
    if arm not in ARMS:raise ValueError('invalid arm')
    if arm=='original_bf16':
        linear=lambda z,w:ledger.call('bf16_linear',api.bf16_linear,z,w)
    elif arm=='dense_group16':
        def linear(z,w):
            a,scale=ledger.call('dense16_prepare',api.dense16_prepare,z,w.input_global_scale)
            return ledger.call('dense16_gemm',api.dense16_gemm,a,w,scale)
    elif arm=='dense_shared32':
        linear=lambda z,w:dense32_linear(z,w,api,ledger)
    else:
        linear=lambda z,w:sharq_linear(z,w,api,ledger)
    pre=linear(x,first)
    h=ledger.call('gelu_tanh',api.gelu_tanh,pre)
    y=linear(h,second)
    gated=ledger.call('gate_multiply',api.gate_multiply,y,gate)
    return {'pre_gelu':pre,'h':h,'y':y,'gated':gated}


def expected_calls(arm):
    common={'gelu_tanh':1,'gate_multiply':1}
    if arm=='original_bf16':return {**common,'bf16_linear':2}
    if arm=='dense_group16':return {**common,'dense16_prepare':2,'dense16_gemm':2}
    if arm=='dense_shared32':return {**common,'dense32_prepare':2,'output_scale':2,'dense32_gemm':2,'bias_add':2}
    if arm=='sharq':return {**common,'sharq_prepare':2,'output_scale':2,'sparse_gemm':2,'dense_residual_gemm':2,'bias_add':2}
    raise ValueError('invalid arm')
