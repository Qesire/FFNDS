"""Finite-population ordered-pair support transfer under uniform row sampling."""
import numpy as np


def pair_populations(total,tile,frame):
    if not all(type(v) is int and v>0 for v in [total,tile,frame]) or total%frame:
        raise ValueError('Positive integer native rows/tile/frame with complete frames required')
    within=within_frame=0
    for start in range(0,total,tile):
        end=min(start+tile,total);length=end-start
        within+=length*(length-1)
        cursor=start
        while cursor<end:
            boundary=min(end,(cursor//frame+1)*frame);segment=boundary-cursor
            within_frame+=segment*(segment-1);cursor=boundary
    sameframe=(total//frame)*frame*(frame-1)
    return {'within_tile':within,'other_tile_same_frame':sameframe-within_frame,
        'all_other_tiles':total*(total-1)-within}


def transfer_statistics(profiles,rows,total,tile,frame,static_groups,keep):
    p=np.asarray(profiles,dtype=np.float64);ids=np.asarray(rows)
    static=np.asarray(static_groups)
    if (p.ndim!=2 or len(ids)!=len(p) or len(p)<2 or ids.ndim!=1 or ids.dtype.kind not in 'iu'
            or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=total)
            or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1.,rtol=0,atol=1e-12)
            or not 0<keep<=p.shape[1] or static.shape!=(keep,) or static.dtype.kind not in 'iu'
            or len(np.unique(static))!=keep or np.any(static<0) or np.any(static>=p.shape[1])):
        raise ValueError('Finite normalized profiles and distinct valid sampled coordinates/support required')
    populations=pair_populations(total,tile,frame)
    selected=np.argsort(-p,axis=1,kind='stable')[:,:keep]
    masks=np.zeros_like(p);np.put_along_axis(masks,selected,1.,axis=1)
    transfer=p@masks.T
    fixed=p[:,static].sum(1)
    gain=transfer-fixed[:,None]
    distinct=~np.eye(len(p),dtype=bool)
    same_tile=ids[:,None]//tile==ids[None,:]//tile
    same_frame=ids[:,None]//frame==ids[None,:]//frame
    subsets={'within_tile':distinct&same_tile,'other_tile_same_frame':distinct&same_frame&~same_tile,
        'all_other_tiles':distinct&~same_tile}
    probability=len(p)*(len(p)-1)/(total*(total-1))
    result={'sample_rows':len(p),'full_rows':total,'tile':tile,'frame_rows':frame,
        'ordered_pair_inclusion_probability':probability,'row_static_C_mean':float(fixed.mean()),
        'row_oracle_C_mean':float(np.diag(transfer).mean()),'subsets':{}}
    for name,mask in subsets.items():
        count=int(mask.sum());population=populations[name]
        observed=float(gain[mask].sum());retained=float(transfer[mask].sum())
        result['subsets'][name]={'sample_ordered_pairs':count,'population_ordered_pairs':population,
            'HT_gain_total':observed/probability,'HT_gain_mean':observed/probability/population if population else None,
            'HT_retained_mean':retained/probability/population if population else None,
            'sample_ratio_gain_mean':observed/count if count else None,
            'sample_ratio_retained_mean':retained/count if count else None}
    return result
