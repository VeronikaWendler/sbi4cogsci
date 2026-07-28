
n=100   # total number of coin flips
p=.6   # probability of single-trial success

# generate some random data
data=rbinom(n,size=1,prob=p)
x=sum(data)

################################################################# prior

alpha=2    # first parameter
beta=4     # second parameter 

############################################################################### let's do some inference

#  set parameters, specify functions, and initialize storage objects
eps=0  	  				# tolerance threshold
N=1000							# number of particles
theta=score=numeric(N)					# declare a vector for storage

rho=function(x,y,n.trials) abs(sum(x)-sum(y))/n.trials			# rho function

# specify bounds for bias parameter
p.upper=1
p.lower=0

##############################################  run Algorithm 1 

for(i in 1:N){  					# loop over particles
  d=eps+1						# initialize d to be greater than tolerance threshold
  while(d>eps) {					# continue proposal generation until tolerance condition is satisfied
    theta.1=runif(1,p.lower,p.upper)			# sample from prior
    x=rbinom(n,1,theta.1)				# simulate data
    d=rho(data,x,n)					# compute distance
  }
  theta[i]=theta.1					# store the accepted value
  score[i]=(sum(data)-sum(x))/n
  if(i%%100==0)print(paste("Simulation ",round(i/N*100),"% Complete.",sep=""))
}

############################################################################### plotting

par(mfrow=c(1,2))

dat=mean(data)
plot(score,theta,ylab="p",xlab="Distance",col=rgb(0,0,1,.3))
abline(v=c(-1,1)*eps,col="green",lwd=5)
abline(v=0,lty=2,lwd=5)
points(0,dat,pch=4,col="red",lwd=10,cex=4)

truepost=dbeta(xs,alpha+sum(data),beta+n-sum(data))
ylim=c(0,max(truepost))

hist(theta,xlab="p",prob=T,breaks=20,main="",ylim=ylim)  
abline(v=p,col="red",lty=1, lwd=5)			# true parameter value that generated the data

lines(ps,truepost,type="l",col="red")
lines(xs,dbeta(xs,alpha,beta),lwd=3,col="blue")

