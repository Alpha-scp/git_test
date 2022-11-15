import numpy as np
import numpy.fft as nf
import matplotlib.pylab as plt
from scipy import signal

fs=1000
f=open('.\ECG_1000Hz_64.dat','r')
data=[]
for i in f:
    s=i.strip()
    data.append(float(s))
f.close()
M=1001
num=np.size(data)
data=np.array(data)
time=np.arange(0,num/1000,0.001)

plt.figure(1)
plt.title('Ogiginal data Time Domain',fontsize=16)
plt.xlabel('Time',fontsize=12)
plt.ylabel('Signal',fontsize=12)
plt.grid(linestyle=':')
plt.plot(time,data,color='dodgerblue',label='Signal')
plt.legend()
plt.savefig('time.jpg')

freqs = nf.fftfreq(round(data.size), 1/fs)
comple = nf.fft(data)
pows = comple*np.conj(comple)
plt.figure(2)
plt.title('Ogiginal data Frequence Domain',fontsize=16)
plt.xlabel('Freq',fontsize=12)
plt.ylabel('Power',fontsize=12)
plt.grid(linestyle=':')
plt.semilogy(freqs[freqs>0],pows[freqs>0],color='dodgerblue',label='freq')
plt.legend()
plt.savefig('freq.jpg')

def highpassDesign(cutoff_freq):
    k=int(cutoff_freq/fs*M)
    X=np.ones(M)
    X[0:k]=0
    X[M-k:M]=0
    x=np.fft.ifft(X)
    x=np.real(x)
    h=np.zeros(M)
    h[0:int(M/2)]=x[int(M/2):M-1]
    h[int(M/2):M-1]=x[0:int(M/2)]
    return h

def bandstopDesign(cutoff_freq1,cutoff_freq2):
    k1=int(cutoff_freq1/fs*M)
    k2=int(cutoff_freq2/fs*M)
    X=np.ones(M)
    X[k1:k2+1]=0
    X[M-k2:M-k1+1]=0
    x=np.fft.ifft(X)
    x=np.real(x)
    h=np.zeros(M)
    h[0:int(M/2)]=x[int(M/2):M-1]
    h[int(M/2):M-1]=x[0:int(M/2)]
    return h


class FIRfilter:
    def __init__(self,_coef1,_coef2):
        self.highpass=_coef1*np.hamming(1001)
        self.bandstop=_coef2*np.hamming(1001)
        self.signal=[]
        self.signal_pos=-1
        self.history=[]
        self.middle=[]
        
    def signal_refresh(self):
        if(self.signal_pos==-1):
            for i in range(M):
                self.signal.append(0)
                self.middle.append(0)
            self.signal[0]=data[0]
            self.signal_pos=1 
        else:
            self.signal.pop(1000)
            self.middle.pop(1000)
            self.signal.insert(0,data[self.signal_pos])
            self.signal_pos=self.signal_pos+1
        return 0
    
    def doConvolution1(self,x,h):
        s=0
        index=self.signal_pos
        for i in range(len(x)):
            s=s+x[i]*h[i]
            index=index-1
            if(index<=0):
                break
        self.middle.insert(0,s)   
        return s
    
    def doConvolution2(self,x,h):
        s=0
        index=self.signal_pos
        for k in range(M):
            s = s + h[k]*x[k]
            index = index-1
            if(index<=0):
                break
        self.history.append(s)
        return s
    
    def dofilter(self):
        self.signal_refresh()
        y=self.doConvolution1(self.signal, self.highpass)
        y=self.doConvolution2(self.middle, self.bandstop)
        #time=np.arange(0,len(self.history)/1000,0.001)
        #plt.plot(time,self.history)
        return y


fir=FIRfilter(highpassDesign(2),bandstopDesign(45,55))
for i in range(30000):
    fir.dofilter()
    #plt.pause(0.1)
    
plt.figure(3)
plt.title('Filtered data Time Domain',fontsize=16)
plt.xlabel('Time',fontsize=12)
plt.ylabel('Signal',fontsize=12)
plt.grid(linestyle=':')
plt.plot(time,fir.history,color='dodgerblue',label='Filtered Signal')
plt.legend()
plt.savefig('filtered_time.jpg')

plt.figure(4)
filtered_comple = nf.fft(fir.history)
filtered_pow=filtered_comple*np.conj(filtered_comple)
plt.title('Filtered data Frequency Domain',fontsize=16)
plt.xlabel('Freq',fontsize=12)
plt.ylabel('Power',fontsize=12)
plt.grid(linestyle=':')
plt.semilogy(freqs[freqs>0],filtered_pow[freqs>0],color='dodgerblue',label='Filtered Signal')
plt.legend()
plt.savefig('filtered_freq.jpg')



"""
h=highpassDesign(2)
h=h*np.blackman(M)
ban=bandstopDesign(45,55)
ban=ban*np.blackman(M)
y2=signal.lfilter(h,1,data)
y3=signal.lfilter(ban,1,y2)
plt.plot(time,y3,alpha=0.5,label='filted')
plt.legend()
plt.show()        
"""   
        