import cupy as cp
from scipy.special import ndtr
from scipy.cluster.vq import kmeans2


def coarse_grain(data, scale):
    x_len, y_len = data.shape
    x_len_coarse = x_len // scale
    y_len_coarse = y_len // scale
    data_coarse = cp.zeros((x_len_coarse, y_len_coarse))
    
    for i in range(x_len_coarse):
        for j in range(y_len_coarse):
            start_x, end_x = i*scale, (i+1)*scale
            start_y, end_y = j*scale, (j+1)*scale
            block = data[start_x:end_x, start_y:end_y]
            block_avg = cp.mean(block)
            data_coarse[i, j] = block_avg
    return data_coarse


def cupy_unique_axis0(array):
    sortarr = array[cp.lexsort(array.T[::-1])]
    mask = cp.empty(array.shape[0], dtype=cp.bool_)
    mask[0] = True
    mask[1:] = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]

def DispEn2D(Mat, m=None, tau=1, c=3, Typex='NCDF', Logx=cp.exp(1), Norm=False, Lock=True):
    Mat = cp.squeeze(Mat)
    NL,NW = Mat.shape    
    if m is None:
        m = cp.array(Mat.shape)//10
    if Logx == 0:
        Logx = cp.exp(1)
           
    if isinstance(m,int):
        mL = int(m); mW = int(m)        
    else:
        mL = int(m[0]); mW = int(m[1])   
        
    if Typex.lower() == 'linear':
        Zi = cp.digitize(Mat,cp.arange(cp.min(Mat),cp.max(Mat),cp.ptp(Mat)/c))
        
    elif Typex.lower() == 'kmeans':        
        Clux, Zx = kmeans2(Mat.flatten(), c, iter=200)
        Zx += 1;  xx = cp.argsort(Clux) + 1;     Zi = cp.zeros(Mat.size);
        for k in range(1,c+1):
            Zi[Zx==xx[k-1]] = k
        Zi = cp.reshape(Zi,Mat.shape)
        
        del Zx, Clux, xx
                        
    elif Typex.lower() == 'ncdf':  
        Zx = ndtr((Mat-cp.mean(Mat))/cp.std(Mat))
        Zi = cp.digitize(Zx,cp.arange(0,1,1/c))  
        
        del Zx
                        
    elif Typex.lower() == 'equal':
        ix = cp.argsort(Mat.flatten())
        xx = cp.round(cp.arange(0,2*Mat.size,Mat.size/c)).astype(int)
        Zi = cp.zeros(Mat.size)
        for k in range(c):
            Zi[ix[xx[k]:xx[k+1]]] = k+1 
            
        Zi = cp.reshape(Zi, Mat.shape)
        del ix, xx, k                   
              
    NL -= (mL - 1)*tau
    NW -= (mW - 1)*tau
    X = cp.zeros((NL*NW,mL*mW))
    p = 0
    for k in range(0,NL):        
        for n in range(0,NW):
              X[p,:] = Mat[k:mL*tau+k:tau,n:mW*tau+n:tau].flatten('F')
              p += 1              
    if p != NL*NW:
        print('Warning: Potential error with submatrix division.')        
    T  = cupy_unique_axis0(X)
    Nx = T.shape[0]
    Counter = cp.sum(cp.all(X[:, None] == T, axis=2), axis=0)
    Ppi = Counter[Counter!= 0]/X.shape[0]
    
    RDE_val = (Ppi - (1/(c**(mL*mW))))**2
    RDE = RDE_val.sum()
    
    Disp2D = -(Ppi*cp.log(Ppi)/cp.log(Logx)).sum()
    
    if round(cp.sum(Ppi).item()) != 1:
        print('Potential error calculating probabilities')          
    if Norm:
        Disp2D /= (cp.log(c**(mL*mW))/cp.log(Logx))
        RDE /= (1 - (1/(c**(mL*mW))))     
    
    del Mat, Zi, X, T, Counter, Ppi

    return Disp2D, RDE

def DistEn2D(Mat, m=None, tau=1, Bins='Sturges', Logx=2, Norm=2, Lock=True):
    Mat = cp.squeeze(Mat)
    NL,NW = Mat.shape    
    if m is None:
        m = cp.array(Mat.shape)//10
    if Logx == 0:
        Logx = cp.exp(1)
           
    
    if isinstance(m,int):
        mL = int(m); mW = int(m)        
    else:
        mL = int(m[0]); mW = int(m[1])   
           
    if Norm == 1 or Norm == 2:
        Mat = (Mat - cp.min(Mat))/cp.ptp(Mat[:])
   
    NL = NL - (mL - 1)*tau
    NW = NW - (mW - 1)*tau
    X = cp.zeros((NL*NW,mL,mW))
    p = 0
    for k in range(0,NL):        
        for n in range(0,NW):
              X[p,:,:] = Mat[k:mL*tau+k:tau,n:mW*tau+n:tau]
              p += 1              
    if p != NL*NW:
        print('Warning: Potential error with submatrix division.')        
    Ny = int(p*(p-1)/2)
    if Ny > 300000000:
        print('Warning: Number of pairwise distance calculations is ' + str(Ny))
    del Mat, NL, NW
    Y = cp.zeros(Ny)
    for k in range(1,p):
        Ix = (int((k-1)*(p - k/2)), int(k*(p-((k+1)/2))))        
        Y[Ix[0]:Ix[1]] = cp.max(cp.abs(X[k:,:,:] - X[k-1,:,:]),axis=(1,2))
     
    if isinstance(Bins, str):
        if Bins.lower() == 'sturges':
            Bx = cp.ceil(cp.log2(Ny) + 1)
        elif Bins.lower() == 'rice':
            Bx = cp.ceil(2*(Ny**(1/3)))
        elif Bins.lower() == 'sqrt':
            Bx = cp.ceil(cp.sqrt(Ny))
        else:
            raise Exception('Please enter a valid binning method')               
    else:
        Bx = Bins
        
    By = cp.linspace(cp.min(Y),cp.max(Y),int(Bx+1))
    Ppi,_ = cp.histogram(Y,By)        
    Ppi = Ppi/Ny
    if round(float(cp.sum(Ppi)),6) != 1:
        print('Warning: Potential error estimating probabilities (p = ' +str(cp.sum(Ppi))+ '.')
        Ppi = Ppi[Ppi!=0]
    elif any(Ppi==0):
        print('Note: '+str(cp.sum(Ppi==0))+'/'+str(len(Ppi))+' bins were empty')
        Ppi = Ppi[Ppi!=0]
           
    Dist2D = -cp.sum(Ppi*cp.log(Ppi)/cp.log(Logx))
    if Norm >= 2:
        Dist2D = Dist2D/(cp.log(Bx)/cp.log(Logx))
    del m, tau, Bins, Logx, Norm, Lock
    del X, p, Ny, Y, Bx, By, Ppi
    return Dist2D

def EspEn2D(Mat, m=None, tau=1, r=20, ps=0.7, Logx=cp.exp(1), Lock=True):
    Mat = cp.squeeze(Mat)
    NL,NW = Mat.shape    
    if m is None:
        m = cp.array(Mat.shape)//10
           
    
    if isinstance(m,int):
        mL = int(m); mW = int(m)        
    else:
        mL = int(m[0]); mW = int(m[1])       
        
    NL = NL - (mL-1)*tau
    NW = NW - (mW-1)*tau
    X = cp.zeros((NL*NW,mL,mW))
    p = 0
    for k in range(NL):        
        for n in range(NW):
            X[p,:,:] = Mat[k:(mL*tau)+k:tau,n:(mW*tau)+n:tau]
            p += 1              
            
    if p != NL*NW:
        print('Warning: Potential error with submatrix division.')        
    Ny = int(p*(p-1)/2)
    if Ny > 300000000:
        print('Warning: Number of pairwise distance calculations is ' + str(Ny))
    Cij = -cp.ones((p-1,p-1))
    for k in range(p-1):
        Temp = abs(X[k+1:,:,:] - X[k,:,:]) <= r
        Cij[:p-(k+1),k] = cp.sum(Temp,axis=(1,2))
     
    Dm = cp.sum((Cij/(mL*mW))>=ps)/(p*(p-1)/2)
    Esp2D = -cp.log(cp.sum(Dm))/cp.log(Logx)
    del Mat, NL, NW, m, tau, r, ps, Logx, Lock
    del X, p, Ny, Cij, Temp, Dm
    return Esp2D
    
def default(x,r):   
    assert isinstance(r,tuple), 'When Fx = "Default", r must be a two-element tuple.'
    y = cp.exp(-(x**r[1])/r[0])
    return y

def FuzzEn2D(Mat, m=None, tau=1, r=None, Fx='default', Logx=cp.exp(1), Lock=True):
    Mat = cp.squeeze(Mat)
    NL,NW = Mat.shape     
    if m is None:
        m = cp.array(Mat.shape)//10
    r = (r, 2)

    
    if isinstance(m,int):
        mL = int(m); mW = int(m)        
    else:
        mL = int(m[0]); mW = int(m[1])   
    
    if isinstance(r,tuple) and Fx.lower()=='linear':
        r = 0
        print('Multiple values for r entered. Default value (0) used.') 
    elif isinstance(r,tuple) and Fx.lower()=='gudermannian':
        r = r[0];
        print('Multiple values for r entered. First value used.')     
    Fun = globals()[Fx.lower()]
    NL = NL - mL*tau
    NW = NW - mW*tau
    X1 = cp.zeros((NL*NW,mL,mW))
    X2 = cp.zeros((NL*NW,mL+1,mW+1))
    p = 0
    for k in range(NL):        
        for n in range(NW):
            Temp2 = Mat[k:(mL+1)*tau+k:tau,n:(mW+1)*tau+n:tau]
            Temp1 = Temp2[:-1,:-1]
            X1[p,:,:] = Temp1 - cp.mean(Temp1)
            X2[p,:,:] = Temp2 - cp.mean(Temp2)
            p += 1            
    if p != NL*NW:
        print('Warning: Potential error with submatrix division.')        
    Ny = int(p*(p-1)/2)
    if Ny > 300000000:
        print('Warning: Number of pairwise distance calculations is ' + str(Ny))
    del Mat, NL, NW
    Y1 = cp.zeros(p-1)
    Y2 = cp.zeros(p-1)
    for k in range(p-1):
        Temp1 = Fun(cp.max(cp.abs(X1[k+1:,:,:] - X1[k,:,:]),axis=(1,2)),r)
        Y1[k] = cp.sum(Temp1)        
        Temp2 = Fun(cp.max(cp.abs(X2[k+1:,:,:] - X2[k,:,:]),axis=(1,2)),r)
        Y2[k] = cp.sum(Temp2) 
        
    Fuzz2D = -cp.log(cp.sum(Y2)/cp.sum(Y1))/cp.log(Logx)
    del m, tau, r, Fx, Logx, Lock
    del X1, X2, p, Ny, Y1, Y2
    return Fuzz2D

def SampEn2D(Mat, m=None, tau=1, r=None, Logx=cp.exp(1), Lock=True):           
    Mat = cp.squeeze(Mat)
    NL,NW = Mat.shape    
    if m is None:
        m = cp.array(Mat.shape)//10
    if r == None:
        r = 0.2*cp.std(Mat)
    
    if isinstance(m,int):
        mL = int(m); mW = int(m)        
    else:
        mL = int(m[0]); mW = int(m[1])       
        
    NL = NL - mL*tau
    NW = NW - mW*tau
    X = cp.zeros((NL*NW,mL+1,mW+1))
    p = 0
    for k in range(NL):        
        for n in range(NW):
            X[p,:,:] = Mat[k:(mL+1)*tau+k:tau,n:(mW+1)*tau+n:tau]
            p += 1              
            
    if p != NL*NW:
        print('Warning: Potential error with submatrix division.')        
    Ny = int(p*(p-1)/2)
    if Ny > 300000000:
        print('Warning: Number of pairwise distance calculations is ' + str(Ny))
    del Mat, NL, NW
    Y1 = cp.zeros(p-1)
    Y2 = cp.zeros(p-1)
    for k in range(p-1):
        Temp = (cp.max(cp.abs(X[k+1:,:-1,:-1] - X[k,:-1,:-1]),axis=(1,2)) < r)
        Y1[k] = cp.sum(Temp)        
        Temp = cp.where(Temp==True)[0] + k + 1
        Y2[k] = cp.sum(cp.max(cp.abs(X[Temp,:,:] - X[k,:,:]),axis=(1,2)) < r)
     
    Phi1 = cp.sum(Y1)/Ny
    Phi2 = cp.sum(Y2)/Ny
    SE2D = -cp.log(Phi2/Phi1)/cp.log(Logx)
    del m, tau, r, Logx, Lock
    del X, p, Ny, Y1, Y2
    return SE2D, (Phi1,Phi2)

def PermEn2D(Mat, m=None, tau=1, Norm=True, Logx=cp.exp(1), Lock=True):            
    Mat = cp.squeeze(Mat)
    NL,NW = Mat.shape    
    if m is None:
        m = cp.array(Mat.shape)//10
           
    
    if isinstance(m,int):
        mL = int(m); mW = int(m)        
    else:
        mL = int(m[0]); mW = int(m[1])       
        
    NL = NL - (mL-1)*tau
    NW = NW - (mW-1)*tau
    
    Temp = Mat[:(mL-1)*tau+1:tau,:(mW-1)*tau+1:tau]
    Temp_flat = Temp.flatten('F')
    Sord = cp.sort(Temp_flat)
    Dict = cp.argsort(Temp_flat, kind='stable')
    if cp.any(cp.diff(Sord)==0):
        for x in cp.where(cp.diff(Sord)==0)[0]+1:
            Dict[x] = Dict[x-1] 
                    
    Counter = 0
    Dict = cp.expand_dims(Dict,axis=0)
    
    for k in range(NL):        
        for n in range(NW):            
            Temp = Mat[k:(mL-1)*tau+k+1:tau,n:(mW-1)*tau+n+1:tau]
            Temp_flat = Temp.flatten('F')
            Sord = cp.sort(Temp_flat)
            Dx = cp.argsort(Temp_flat,kind='stable')
        
            if cp.any(cp.diff(Sord)==0):
                for x in cp.where(cp.diff(Sord)==0)[0]+1:
                    Dx[x] = Dx[x-1] 
                    
            if cp.any(~cp.any(Dict - Dx, axis=1)):
                Counter += ~cp.any(Dict - Dx, axis=1)*1
            
            else:
                Dict = cp.vstack((Dict, Dx))
                Counter = cp.hstack((Counter, 1))
                
    if cp.sum(Counter) != NL*NW:
        print('Warning: Potential error with permutation comparisons.')        

    Pi = Counter/cp.sum(Counter) 
    Perm2D = -cp.sum(Pi*cp.log(Pi)/cp.log(Logx))
                  
    if Norm:
        Perm2D = Perm2D/(cp.log(float(cp.math.factorial(mL*mW)))/cp.log(Logx))
    del Mat, NL, NW, m, tau, Norm, Logx, Lock
    del Temp, Sord, Dict, Counter
    del Pi              
    return Perm2D
