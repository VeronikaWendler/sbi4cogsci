
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

log.like=function(x,times,data,func){
  pred=func(times,x)
  dens=numeric(length(times))
  for(i in 1:length(times)){
    dens[i]=sum(dbinom(sum(data[,i]),n.trials,pred[i],log=TRUE))
  }
  out=sum(dens)
  out
}

# generate data from one model, both training and test data
x=c(.5,1.3,.2)
pred=m2(times,x[1:3])
train.data=sapply(pred,rbinom,size=1,n=n.trials)
test.data=sapply(pred,rbinom,size=1,n=n.trials)

par(mfrow=c(1,1))
plot(times,pred,ylab="Probability of Recall",xlab="Time",col=colset[2],type="b",pch=16)
lines(times,apply(train.data,2,mean),col="forest green",type="p",pch=16)
lines(times,apply(test.data,2,mean),col="lime green",type="p",pch=16)

# fit the three models to the training data, obtain model parameters
fit1=optim(c(x[1]),log.like,times=times,func=m1,data=train.data,control=list("fnscale"=-1))
fit2=optim(c(x[1:2]),log.like,times=times,func=m2,data=train.data,control=list("fnscale"=-1))
fit3=optim(c(x[1:3]),log.like,times=times,func=m3,data=train.data,control=list("fnscale"=-1))
c(fit1$value,fit2$value,fit3$value)

# now make predictions for the test data
pred1=m1(times,fit1$par)
pred2=m2(times,fit2$par)
pred3=m3(times,fit3$par)

# first, plot the obtained fits on the training data
colset=c("black","red","blue")
plot(times,pred1,ylab="Probability of Recall",xlab="Time",col=colset[1],type="b",pch=16)
lines(times,pred2,col=colset[2],type="b",pch=16)
lines(times,pred3,col=colset[3],type="b",pch=16)
lines(times,apply(train.data,2,mean),col="forest green",type="p",pch=16)

# now let's see how accurate these predictions are for test data
mean.test.data=apply(test.data,2,mean)
plot(times,pred1,ylab="Probability of Recall",xlab="Time",col=colset[1],type="b",pch=16)
lines(times,pred2,col=colset[2],type="b",pch=16)
lines(times,pred3,col=colset[3],type="b",pch=16)
lines(times,mean.test.data,col="lime green",type="p",pch=16)

sse1=sqrt(sum((mean.test.data-pred1)^2))
sse2=sqrt(sum((mean.test.data-pred2)^2))
sse3=sqrt(sum((mean.test.data-pred3)^2))
c(sse1,sse2,sse3)
  
  