
m1=function(t,x)(1+t)^(-x[1])
m2=function(t,x)(x[2]+t)^(-x[1])
m3=function(t,x)(x[2]+x[3]*t)^(-x[1])

times=c(0.1, 2.1, 4.1, 6.1, 8.1)

x=c(.5,1.3,.2)
colset=c("black","red","blue")
plot(times,m1(times,x[1]),ylab="Probability of Recall",xlab="Time",col=colset[1],type="b",pch=16)
lines(times,m2(times,x[1:2]),col=colset[2],type="b",pch=16)
lines(times,m3(times,x),col=colset[3],type="b",pch=16)

################################################################ simulation

log.like=function(x,times,data,func=m3){
  pred=func(times,x)
  dens=numeric(length(times))
  for(i in 1:length(times)){
    dens[i]=sum(dbinom(sum(data[,i]),n.trials,pred[i],log=TRUE))
  }
  out=sum(dens)
  out
}

n.reps=100 # how many simulations to run?
# need a 3x3xreps matrix for storage
bic=sse=aic=array(NA,c(3,3,n.reps))
n.trials=50 # how many trials per time point?

for(q in 1:n.reps){
for(i in 1:3){
# first, generate data from model i
  if(i==1)pred=m1(times,x[1:i])
  if(i==2)pred=m2(times,x[1:i])
  if(i==3)pred=m3(times,x[1:i])
  data=sapply(pred,rbinom,size=1,n=n.trials)
    for(j in 1:3){
      # next, fit model j to data from model i
      if(j==1)func=m1
      if(j==2)func=m2
      if(j==3)func=m3
      fit=optim(c(x[1:j]),log.like,times=times,func=func,data=data,control=list("fnscale"=-1))
      k=length(x[1:j]) # number of free parameters
      n=n.trials*length(times) # total number of observations
      aic[i,j,q]=-2*fit$value + 2*k
      bic[i,j,q]=-2*fit$value + k*log(n)
      mean.data=apply(data,2,mean)
      preds=func(times,x[1:j])
      #preds=func(times,fit$par)
      sse[i,j,q]=sqrt(sum((mean.data-preds)^2))
  }
}
  print(paste("Simulation",round(q/n.reps*100),"% complete."))
}

out.bic=out.sse=out.aic=matrix(0,3,3)
for(i in 1:n.reps){
temp=apply(bic[,,i],1,which.min)
for(j in 1:3)out.bic[j,temp[j]]=out.bic[j,temp[j]]+1
temp=apply(aic[,,i],1,which.min)
for(j in 1:3)out.aic[j,temp[j]]=out.aic[j,temp[j]]+1
temp=apply(sse[,,i],1,which.min)
for(j in 1:3)out.sse[j,temp[j]]=out.sse[j,temp[j]]+1
}

out.bic
out.aic
out.sse


