
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

############################################################################### let's do some inference

#  set parameters, specify functions, and initialize storage objects
eps=0			  				    # tolerance threshold
tol=.1                  # acceptable window
N=1000	  						    # number of samples
theta=matrix(NA,N,2)			# declare a vector for storage (remember there are two parameters)

#rho=function(x,y) sqrt((x$hr-y$hr)^2 + (x$far-y$far)^2)			# one rho function
rho=function(x,y,tol=.1) (abs(x$hr-y$hr)>tol) + (abs(x$far-y$far)>tol)  		# another rho function

# specify bounds for bias parameter
bias.upper=1
bias.lower=-1

# specify bounds for discriminability parameter
disc.upper=4
disc.lower=0

##############################################  run Algorithm 1 

for(i in 1:N){						# loop over particles
  d=eps+1						# initialize d to be greater than tolerance threshold
  while(d>eps) {					# continue proposal generation until tolerance condition is satisfied
    theta.1=runif(1,bias.lower,bias.upper)			# sample from prior
    theta.2=runif(1,disc.lower,disc.upper)  		# sample from prior
    x=raw_sdt(n.trials,n.signal,disc=theta.2,theta.1)				# simulate data
    d=rho(data,x,tol)					# compute distance
  }
  theta[i,]=c(theta.1,theta.2)					# store the accepted value
  if(i%%100==0)print(paste("Simulation ",round(i/N*100),"% Complete.",sep=""))
}

############################################################################### plotting

lwd=4
breaks=30

par(mfrow=c(2,2),cex=1)

far=data$far
hr=data$hr
plot(NA,xlim=c(0,1),ylim=c(0,1),xlab="False Alarm Rate", ylab="Hit Rate")
abline(0,1,lwd=lwd)
lines(c(.5,-1),c(.5,2),lwd=lwd)
xleft=far-tol
xright=far+tol
ybottom=hr-tol
ytop=hr+tol
rect(xleft,ybottom,xright,ytop,lwd=6,col="green")
points(far,hr,pch=4,cex=2.5,lwd=10,col="red")

plot(NA,xlim=c(0,3),ylim=c(-1,1),xlab="Discriminability", ylab="Bias")
points(theta[,2],theta[,1],col=rgb(0,0,1,.3))
points(disc,crit-disc/2,pch=16,cex=2)

hist(theta[,1],main="",xlab="Bias",ylab="Density",breaks=breaks,prob=TRUE)
abline(v=crit-disc/2,lwd=lwd,col="red")

hist(theta[,2],main="",xlab="Discriminability",ylab="Density",breaks=breaks,prob=TRUE)
abline(v=disc,lwd=lwd,col="red")

