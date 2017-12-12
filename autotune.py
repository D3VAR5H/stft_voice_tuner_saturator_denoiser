"""A dirty hack to denoise, autotune and saturate voice"""

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
import sys, random, struct
from copy import copy
from math import floor,ceil,pi,sin,cos,hypot,asin,atan,exp
from scipy import signal,interpolate,fftpack
from operator import attrgetter
#import png
import numpy as np
import matplotlib.pyplot as plt
yr=4096
ffb=8192
yr2 = (ffb>>1)+1
nps=ffb
nol=nps-nps//4
win='blackmanharris'
sr=48000#change to 32000 if you want Virtual ANS compatibility
notemax=127#127#change to 119 if you want Virtual ANS compatibility
freq=sr>>1
freqarray=np.asarray([(2**((nt/(yr-1)*notemax-57)/12))*440 for nt in range(yr-1,-1,-1)],dtype=np.float32)
binarray=np.asarray([nt/(yr2-1)*freq for nt in range(yr2)],dtype=np.float32)
fmm=[nt/(yr-1)*notemax for nt in range(yr-1,-1,-1)]
fmmi=[int((nt/notemax)*(yr-1)+.5) for nt in range(notemax,-1,-1)]#was melarray
smlp=0
cmlp=0
def melPeak(line):
    global cmlp,smlp
    #checking for data
    mn=line.mean()
    rest=np.sum(line-mn)
    if mgn(rest)<1e-17:
        #print(mgn(rest),rest)
        return None
    thresh=line[:int((len(line)-1)*.75)].mean()
    for i,b in zip(range(int((len(line)-1)*.75),-1,-1),line[::-1]):
        if b>=thresh:
            cmlp+=1
            smlp+=b
            return i
    return None
def rootkey(rk):
    return {(n+rk)%12 for n in (0, 2, 4, 5, 7, 9, 11)}
def oct_stats(octave):
    probe = sorted(enumerate(octave), key=lambda t: t[1], reverse=True)
    res = set(range(12))
    for item in probe:
        sub = rootkey(item[0])
        tst = res-sub
        if tst:
            res = tst
            if len(tst) == 1:
                break
    res = random.choice(tuple(res))
    rk = rootkey(res)
    ofx = list(range(12))
    for item in range(12):
        if item not in rk:
            ofx[item] = None
    return res, ofx
def melOctave(lpks):
    global fmm
    octave=[0,0,0,0,0,0,0,0,0,0,0,0]
    for b in lpks:
        if b is not None:
            nt=int(fmm[b]+.5)
            octave[nt%12]+=1
    ocst,ofx=oct_stats(octave)
    return ocst,ofx
def fixVocMels(oclw):
    global fmm,fmmi
    global cmlp,smlp
    ocll=oclw.T
    lpks=[]
    for line in ocll:
        lpk=melPeak(line)
        lpks.append(lpk)
    print('avg peak',smlp/cmlp)
    ocst,ofxs=melOctave(lpks)
    ofx=list(ofxs)
    print(ocst,ofx)
    #here blur
    n=4
    n2=n*n
    kernel=[[1/n2]*n2]
    img2=signal.convolve(oclw,kernel,mode='same').T
    lpks=[]
    for line in img2:
        lpk=melPeak(line)
        lpks.append(lpk)
    del img2
    p=0
    for i,lpk in enumerate(lpks):
        if lpk is not None:
            nt=fmm[lpk]
            o,n=divmod(nt+.5,12)
            pn=ofx[int(n)]
            if pn is None:
               pn=int(n-1)
            if ofxs[int(n)-1] is None:
                ofx[int(n)-1]=pn
            if ofxs[(int(n)+1)%12] is None:
                ofx[(int(n)+1)%12]=pn
            nt=int(o*12+pn)
            f=fmmi[nt]
            df=f-lpk
            if not df:
                continue
            elif df>0:
                p+=1
                ocll[i]=np.roll(ocll[i],df)
                ocll[i,:df]=0
                #print('roll forward',df)
            elif df<0:
                p+=1
                ocll[i]=np.roll(ocll[i],df)
                ocll[i,df:]=0
                #print('roll backwards',df)
            else:
                print('madness')
                continue
    print(p,len(lpks))
    #plt.imshow(img2,cmap='gray')
    #plt.show()
    return ocll.T
################################FFT
def nonlocmax(x,y,z,plane):
    if not 0<y<len(plane):
        return True
    if not 0<x<len(plane[0]):
        return True
    if plane[y][x]>z:
        return True
    return False
def sign(num):
    if not num:
        return 0
    return [-1,1][num>0]
def mgn(nm):
    return (nm.real**2+nm.imag**2)**.5
def peak(m):
    b=m[:len(m)//2]
    b[:50]=0
    b[400:]=0
    c=mgn(b)
    d=list(c)
    e=c.max()
    r=np.zeros(m.shape,dtype=m.dtype)
    if not e:
        return r
    i=d.index(e)
    r[i]=m[i]
    r[-i]=m[-i]
    return r
def make_spectrogram(wave,filesr,noise,noisesr,synthsr,synthfreq):
    global yr2,ffb,win,nps,nol,freqarray
    ffttr=signal.stft(wave,filesr,window='blackmanharris',nperseg=int(nps*filesr/synthsr),noverlap=int(nol*filesr/synthsr),nfft=int(ffb*filesr/synthsr),return_onesided=True)
    ns=signal.stft(noise,noisesr,window='blackmanharris',nperseg=int(nps*noisesr/synthsr),noverlap=int(nol*noisesr/synthsr),nfft=int(ffb*noisesr/synthsr),return_onesided=True)
    nsp=np.log(np.pad(ns[2][:ffttr[2].shape[0]],(0,max(0,ffttr[2].shape[0]-ns[2].shape[0])), 'constant', constant_values=(0, 0))+1e-240)
    nsmn=np.mean(nsp,axis=1)
    im_data=ffttr[2]
    binarray=ffttr[0]
    dur=np.max(ffttr[1])
    im_data[:8]=0
    im_data[:,:2]=0
    im_data[:,-2:]=0
    im_data=np.log(im_data-1e-240)
    im_data+=3
    im_data/=-nsmn[:,None]
    im_data*=8#8
    im_data-=im_data.max()
    spl=interpolate.interp2d(ffttr[1],binarray,im_data.real)
    real_data=spl(np.arange(0,dur,3/100),freqarray)
    spl=interpolate.interp2d(ffttr[1],binarray,im_data.imag)
    im_data=real_data+1j*spl(np.arange(0,dur,3/100),freqarray)
    im_data=np.exp(im_data)+1e-240
    im_data=im_data[::-1]
    im_data=fixVocMels(im_data)
    return im_data
def dynamicIncr(bar):
    bar2=np.zeros(bar.shape,dtype=bar.dtype)
    pointer=1
    while pointer<len(bar)-1:
        if bar[pointer]>=bar[pointer-1] and bar[pointer]>=bar[pointer+1]:
            bar2[pointer]=bar[pointer]
        pointer+=1
    return bar2
def gen_wave(in_data):
    global ffb,win,nps,nol,freqarray,freq,yr,binarray,sr
    im_data=in_data
    im_data=np.log(in_data+1e-239)
    spl=interpolate.interp2d(np.arange(0,im_data.shape[1],1),freqarray,im_data.real)
    real_data=spl(np.arange(0,im_data.shape[1],1*nps/sr/4*100/3),binarray)
    spl=interpolate.interp2d(np.arange(0,im_data.shape[1],1),freqarray,im_data.imag)
    im_data=real_data+1j*spl(np.arange(0,im_data.shape[1],1*nps/sr/4*100/3),binarray)
    im_data-=im_data.max()+2
    im_data=np.exp(im_data)-1e-239
    eq=np.concatenate((np.full(800,1.0),(np.arange(800,4097)-4096)/(800-4097)))**3
    saturation=np.asarray([fftpack.ifft(dynamicIncr(fftpack.fft(b)))*eq for b in im_data.T],dtype=im_data.dtype).T
    im_data=(im_data*3+saturation)/4
    im_data=np.log(im_data+1e-239)
    im_data-=im_data.max()+2
    im_data=np.exp(im_data)-1e-239
    res=signal.istft(im_data,window=win,input_onesided=True,noverlap=nol,nperseg=nps,nfft=ffb)[1]
    return res
################################IO
def load_wav(filename):
    stream=open(filename,"rb")
    if stream.read(4)!=b'RIFF':
        sys.exit('not valid wav format')
    expect=struct.unpack('<I',stream.read(4))[0]
    if stream.read(8)!=b'WAVEfmt\x20':
        sys.exit('failed')
    chunksize=struct.unpack('<I',stream.read(4))[0]
    chunk=stream.read(chunksize)
    fmt=struct.unpack('<H',chunk[0:2])[0]
    ch=struct.unpack('<H',chunk[2:4])[0]
    freq=struct.unpack('<I',chunk[4:8])[0]
    rate=struct.unpack('<I',chunk[8:12])[0]
    ba=struct.unpack('<H',chunk[12:14])[0]
    bps=struct.unpack('<H',chunk[14:16])[0]
    if fmt!=1:
        sys.exit('not a PCM')
    if ch!=1:
        sys.exit('not mono')
    if bps not in (8,16,32):
        sys.exit('bit not supported')
    if rate!=freq*ba or ba!=ch*bps/8:
        sys.exit('file corrupted')
    if stream.read(4)!=b'data':
        sys.exit('failed')
    datasize=struct.unpack('<I',stream.read(4))[0]
    expect-=20+chunksize
    expect=min(expect,datasize)
    if bps!=8:
        data=stream.read(divmod(expect,2)[0]*2)
        expect=len(data)
    else:
        data=stream.read(expect)
    fmt2={8:'<B',16:'<h',32:'<l'}[bps]
    wave=struct.iter_unpack(fmt2, data)
    del data
    fmt3={8:np.uint8,16:np.int16,32:np.int32}[bps]
    wave=np.asarray(list(zip(*wave))[0],dtype=fmt3)
    peakl=np.min(wave)
    peakh=np.max(wave).astype(np.float64)-peakl.astype(np.float64)
    wave=(wave.astype(np.float64)-peakl)/peakh
    return (wave,freq)
def write_wave(filename,wave,sr):
    l=-np.min(wave)
    h=np.max(wave)
    a=max(l,h)
    wave/=a
    wave=wave.real
    wave=(wave*32767).astype(np.int16)
    wave=struct.pack('<'+str(len(wave))+'h',*wave)
    wav=open(filename, "wb")
    wav.write(b'RIFF')
    wav.write(struct.pack('<I',36+len(wave)))
    wav.write(b'WAVEfmt\x20\x10\x00\x00\x00\01\00\01\00')
    wav.write(struct.pack('<I',sr))
    wav.write(struct.pack('<I',sr*2))
    wav.write(struct.pack('<H',2))
    wav.write(struct.pack('<H',16))
    wav.write(b'data')
    wav.write(struct.pack('<I',len(wave)))
    wav.write(wave)
    wav.close()
instrument=make_spectrogram(*load_wav("sng.wav"),*load_wav("sngnoise.wav"),sr,freq)
im_data=instrument
im_data = np.sqrt(im_data.real**2+im_data.imag**2)
im_data-=np.min(im_data)
im_data/=np.max(im_data)
im_data=np.log(im_data+1e-12)
im_data-=np.min(im_data)/3
im_data/=np.max(im_data)
im_data[im_data<=0]=0
im_data-=1
im_data*=np.e
im_data=np.exp(im_data)
im_data-=np.min(im_data)
im_data/=np.max(im_data)
im=(im_data*255).astype(np.uint8)
#file = open('voice.png', 'wb')
#xa, yb = im.shape
#w = png.Writer(yb, xa, greyscale=True)
#w.write(file, im)
#file.close()
#print('wrote png')

def spread(cur,cnt):
    cur/=cnt
    sign=[-1,1][cur>0]
    noise=int(cnt/(1/abs(cur)+1))
    signal=cnt-noise
    ret=[sign]*signal+[-sign]*noise
    return ret
def write_pdm(filename,wave,sr):
    wave=wave.real
    l=-np.min(wave)
    h=np.max(wave)
    a=max(l,h)
    wave/=a
    thresh=1/63
    cur=wave[0]
    cnt=1
    quanted=[]
    for p in wave[1:]:
        if abs(p-(cur/cnt))<thresh:
            cur+=p
            cnt+=1
        else:
            #print(cnt,cur/cnt)
            quanted+=spread(cur,cnt)#[cur/cnt]*cnt
            cur=p
            cnt=1
    write_wave(filename,quanted,sr)
wv=gen_wave(instrument)
write_wave('vcp13.wav',wv,sr)
#write_pdm('vcpdm.wav',wv,sr)
print('wrote wav')
