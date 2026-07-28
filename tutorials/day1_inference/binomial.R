
# take a look at the documentation associated with the binomial distribution
?rbinom

n=10   # total number of coin flips
p=.7   # probability of single-trial success

# generate some random data
data=rbinom(n,size=1,prob=p)
x=sum(data)

################################################################# PDFs

# we can evaluate the probability of any experimental result x
d=5    # number of heads in n flips
dbinom(d,size=n,prob=p)

# and ask, which result is more likely, d1 or d2?
d1=5    # number of heads in n flips
d2=6    # number of heads in n flips
hypothesis=c(d1,d2)
names(hypothesis) <- c("d1","d2")
dens=dbinom(hypothesis,size=n,prob=p)

# we can look at the probability of every outcome (i.e., the PDF) visually
xs=seq(0,n,by=1)
fx=dbinom(xs,size=n,prob=p)
plot(xs,fx,type="h",xlab="Number of Heads",ylab="Density")

################################################################# inference

# define a likelihood function
log.like=function(p,x){
  dens=dbinom(x,size=1,prob=p,log=TRUE)
  sum(dens)
}

# evaluate the likelihood at a single point pstar, given our data (i.e., 'data')
pstar=.5
log.like(pstar,data)

# define a grid to evaluate many likelihoods
ps=seq(0,1,.01)
# notice that this doesn't work
lps=log.like(ps,data)
# but this will
lps=numeric(length(ps))
for(i in 1:length(ps)){
  lps[i]=log.like(ps[i],data)
}
# another approach is...
lps=sapply(ps,log.like,x=data)

# why would the likelihood be -Inf at 0 and 1?

# now we can plot the likelihoods
plot(ps,lps,type="b",xlab="p",ylab="Likelihood",pch=16)

# new block division - switch back to slides
################################################################# maximization

# so what is the most likely value for p?
# here is the rough way:
temp=sort(lps,decreasing=TRUE,index.return=TRUE)
ps[temp$ix[1]]
# a cleaner way is...
ps[which.max(lps)]

# we can try this again with much higher resolution:
# define a grid to evaluate many likelihoods
ps=seq(.05,.95,.1)
lps=sapply(ps,log.like,x=data)

# so what is the most likely value for p?
phat=ps[which.max(lps)]

# now we can plot the likelihoods
xlim=c(.05,.95)
plot(ps,lps,type="b",xlab="p",ylab="Likelihood",pch=16,xlim=xlim)
abline(v=phat,col="red",lwd=4)
# comare the estimate to the MLE
abline(v=mean(data),col="blue",lwd=4)

################################################################# optimization

require(msm)

n.iter=500    # total number of iterations
pset=lpset=numeric(n.iter)
tune=.01

pset[1]=.1
lpset[1]=log.like(pset[1],data)
for(t in 2:n.iter){
  pstar=rtnorm(1,pset[t-1],tune,0,1) # use truncated normal to ensure 0<=p<=1
  lpstar=log.like(pstar,data)
  if(lpstar>lpset[t-1]){ 
    # if it's better, take it
    pset[t]=pstar
    lpset[t]=lpstar
  } else { 
    # if it's worse, reject it
    pset[t]=pset[t-1]
    lpset[t]=lpset[t-1]
  }
}

plot(pset,type="l",xlab="Iteration",ylab="p",ylim=c(0,1))
abline(h=mean(data),lwd=3,col=1)

################################################################# let's get serious about optimization

require(msm)

n.iter=300       # total number of iterations
n.particles=100  # total number of particles
pset=lpset=matrix(NA,n.iter,n.particles)
tune=.1

for(i in 1:n.particles){
pset[1,i]=runif(1,.1,.9)
lpset[1,i]=log.like(pset[1,i],data)
}

for(t in 2:n.iter){
  pstar=rtnorm(n.particles,pset[t-1,],tune,0,1) # use truncated normal to ensure 0<=p<=1
  lpstar=sapply(pstar, log.like, data)
  for(i in 1:n.particles){
  if(lpstar[i]>lpset[t-1,i]){ 
    # if it's better, take it
    pset[t,i]=pstar[i]
    lpset[t,i]=lpstar[i]
  } else { 
    # if it's worse, reject it
    pset[t,i]=pset[t-1,i]
    lpset[t,i]=lpset[t-1,i]
  }
  }
}

matplot(pset,type="l",xlab="Iteration",ylab="p",lty=1,ylim=c(0,1))
abline(h=mean(data),lwd=3,col=1)

################################################################# R core form of optimization

optim(.1,log.like,x=data,control=list("fnscale"=-1))
