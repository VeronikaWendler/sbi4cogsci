
n=100   # total number of coin flips
p=.6   # probability of single-trial success

# generate some random data
data=rbinom(n,size=1,prob=p)
x=sum(data)

################################################################# prior

alpha=2    # first parameter
beta=4     # second parameter 

xs=seq(0,1,.01)
ys=dbeta(xs,alpha,beta)
plot(xs,ys,lwd=5,xlab=expression(p),ylab="Density",xlim=c(0,1),type="l")

################################################################# inference

# define a posterior distribution
posterior=function(p,x,alpha,beta,log=TRUE){
  np=length(p)
  dens=numeric(np)
  for(i in 1:np){
  prior=dbeta(p[i],alpha,beta,log=TRUE) # note that the prior doesn't depend on the data
  like=dbinom(x,size=1,prob=p[i],log=TRUE) 
  dens[i]=sum(prior) + sum(like)
  }
  dens[(is.na(dens))] <- -Inf # note, this is a catch to eliminate NA values (as their density will be zero)
  if(log==TRUE)return(dens)
  if(log==FALSE)return(exp(dens))
}

# we can evaluate the unnormalized posterior for different values of p
pstar=.5
posterior(pstar,data,alpha,beta,log=FALSE)

################################################################# posterior approximation, method 1: grid and integrate functions

# to get the full posterior, we need to evaluate that integral in the denominator
# define a grid to evaluate many points
tol=.01
ps=seq(0,1,tol)
lps=sapply(ps,posterior,x=data,alpha,beta,log=FALSE)
plot(ps,lps,type="b",xlab="p",ylab="Posterior",pch=16)

# to compute the integral over p, we use:
den=integrate(posterior,lower=0,upper=1,x=data,alpha=alpha,beta=beta,log=FALSE)$value

# another approach is to numerically approximate it via the Riemann sum
# find the area via bins, then multiply bin widths by lengths, and add them up
den2=sum(lps*tol)

plot(ps,lps/den,type="b",xlab="p",ylab="Posterior",pch=16)
lines(ps,lps/den2,type="l",col="red")

# also add the prior to the plot
lines(xs,dbeta(xs,alpha,beta),lwd=3,col="blue")

################################################################# posterior approximation, method 2: Markov chain Monte Carlo

require(msm)

n.iter=1000    # total number of iterations
pset=lpset=numeric(n.iter)
tune=1

# initialize
pset[1]=mean(data)
lpset[1]=posterior(pset[1],data,alpha=alpha,beta=beta,log=TRUE)

# loop
for(t in 2:n.iter){
  pstar=rtnorm(1,mean=pset[t-1],sd=tune,0,1) # non symmetric transition kernel
  lpstar=posterior(pstar,data,alpha=alpha,beta=beta,log=TRUE)
  qdens_den=dtnorm(pstar,mean=pset[t-1],sd=tune,0,1,log=TRUE)
  qdens_num=dtnorm(pset[t-1],mean=pstar,sd=tune,0,1,log=TRUE)
  # exp(log(ab/cd))
  # exp(log(a/c)+log(b/d))
  # exp(log(a)-log(c)+log(b)-log(d))
  a=exp(lpstar-lpset[t-1]+qdens_num-qdens_den) # note that proposal density is in the numerator
  if(runif(1)<a){ 
    # if it's better, take it
    pset[t]=pstar
    lpset[t]=lpstar
  } else { 
    # if it's worse, reject it (with some probability)
    pset[t]=pset[t-1]
    lpset[t]=lpset[t-1]
  }
}

# plot it
par(mfrow=c(1,2))
# this is called a trace plot
plot(pset,type="l",xlab="Iteration",ylab="p",ylim=c(0,1))
# this is the result of the MCMC, collapsed across time
hist(pset,prob=TRUE,breaks=30,xlim=c(0,1))
lines(ps,lps/den2,type="l",col="red")
lines(xs,dbeta(xs,alpha,beta),lwd=3,col="blue")

