
# key SDT function you need
raw_sdt=function(n.trials,n.signal,disc,bias){
n.noise=n.trials-n.signal  # calculate number of noise trials
mean1=0   # what is the mean of the noise distribution?
mean2=disc  # what is the mean of the signal distribution?
crit=disc/2+bias # where is the criterion?
#signal.trial=rbinom(n.trials,1,n.signal/n.trials) # just one way to generate signal.trial
signal.trial=c(rep(1,n.signal),rep(0,n.noise))  # specifies which trials are signal and which are noise
ms=(signal.trial==1)*mean2 + (signal.trial==0)*mean1  # calculate the mean of the stimulus generating distribution
stimuli=rnorm(n.trials,mean=ms,1)   # generate the "assay value" or stimulus from the representation
resp=(stimuli>crit)*1 + (stimuli<=crit)*0   # generate a response from the model
hr=mean(resp[signal.trial==1])    # calculate the hit rate
far=mean(resp[signal.trial==0])   # calculate the false alarm rate
list("stim"=signal.trial, "resp"=resp, "hr"=hr, "far"=far)
}

############################################################################### run an 'experiment'

# specify the parameters of the model here
disc=1    # discriminability
bias=0    # bias
crit=disc/2+bias  # just a reparameterization

# specify the experiment here
n.trials=100    # how many trials in the experiment?
n.signal=50     # how many of those trials are signal? (i.e., must be less than n.trials)

data=raw_sdt(n.trials,n.signal,disc,bias) # generate data

############################################################################### plotting

lwd=4

par(mfrow=c(2,2),cex=1)

plot(NA,xlim=c(0,3),ylim=c(-1,1),xlab="Discriminability", ylab="Bias")
points(disc,crit-disc/2,pch=16,cex=2)

xs=seq(-5,8,.001)
ys=sapply(1:2,function(x,xs,mu,sd)dnorm(xs,mu[x],sd[x]),mu=c(0,disc),sd=c(1,1),xs=xs)
matplot(NA,xaxt="n",yaxt="n",bty="n",xlim=c(-3,5),type="l",ylim=c(0,.5),xlab="Assay Value", ylab="")
lines(xs[xs>crit],ys[xs>crit,2],type="h",col="light blue")
lines(xs[xs>crit],ys[xs>crit,1],type="h",col="blue")
matlines(xs,ys,lwd=6,lty=1,col=1)
lines(c(0,0),c(0,.4),lty=3,lwd=4)
lines(c(disc,disc),c(0,.4),lty=3,lwd=4)
lines(c(crit,crit),c(0,.43),lty=1,lwd=4)

far=data$far
hr=data$hr
plot(NA,xlim=c(0,1),ylim=c(0,1),xlab="False Alarm Rate", ylab="Hit Rate")
abline(0,1,lwd=lwd)
lines(c(.5,-1),c(.5,2),lwd=lwd)
points(far,hr,pch=4,cex=2.5,lwd=10,col="red")

