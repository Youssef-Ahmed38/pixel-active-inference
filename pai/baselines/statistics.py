import math
from scipy.stats import binomtest


def rate(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(k=int(k),n=n,rate=p,wilson95=[max(0.,mid-half),min(1.,mid+half)])


def paired(candidate,full):
    a={r['index']:r for r in candidate};b={r['index']:r for r in full}
    if set(a)!=set(b):raise ValueError('Unpaired episodes')
    for k in a:
        if a[k]['seed']!=b[k]['seed'] or a[k]['disturbance_draw_sha256']!=b[k]['disturbance_draw_sha256']:raise ValueError('Different paired layouts/disturbances')
    wins=sum(a[k]['success'] and not b[k]['success'] for k in a)
    losses=sum(b[k]['success'] and not a[k]['success'] for k in a)
    return dict(candidate_only=int(wins),full_only=int(losses),two_sided_sign_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.)
