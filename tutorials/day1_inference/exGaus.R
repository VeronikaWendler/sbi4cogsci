
#install.packages("gamlss.dist")
require("gamlss.dist")
require("akima")

n=100      # number of trials

mu=300
sigma=35
nu=100     # henceforth, we will assume nu is known

data=rexGAUS(n,mu,sigma,nu)

hist(data,prob=TRUE,xlab="Response Time",ylab="Density",main="")

# overlay with PDF
xs=seq(0,1000,1)
ys=dexGAUS(xs,mu,sigma,nu)
lines(xs,ys)

########################################################### let's do some inspection

log.like=function(x,data,nu){
  dens=dexGAUS(data,mu=x[1],sigma=x[2],nu=nu,log=TRUE)
  sum(dens)
}

# define a grid over mu and sigma
mus=seq(250,330,1)
sigmas=seq(10,100,1)
grid=expand.grid(mus,sigmas)
like=apply(grid,1,log.like,nu=nu,data=data)

mesh=interp(grid[,1],grid[,2],like)
image(mesh,xlab=expression(mu),ylab=expression(sigma))
points(mu,sigma,pch=4,cex=3,lwd=4)

filled.contour(mesh,xlab=expression(mu),ylab=expression(sigma))
# note, you cannot simply use the point command within filled.contour

######### we can also look at conditional distributions
# first, redefine likelihood for convenience
log.like.cond=function(mu,sigma,data,nu){
  dens=dexGAUS(data,mu=mu,sigma=sigma,nu=nu,log=TRUE)
  sum(dens)
}

mus=seq(250,330,1)
sigmas=seq(10,100,1)

# loop though the grid for one parameter by holding the other fixed
ymus=sapply(mus,log.like.cond,sigma=sigma,nu=nu,data=data)
ysigmas=sapply(sigmas,log.like.cond,mu=mu,nu=nu,data=data)

# plot 'em
par(mfrow=c(1,2))
plot(mus,ymus,xlab=expression(mu),ylab="Likelihood")
abline(v=mu,col="red",lwd=3)
plot(sigmas,ysigmas,xlab=expression(sigma),ylab="Likelihood")
abline(v=sigma,col="red",lwd=3)

################################################################# R core form of optimization

optim(c(250,60),log.like,data=data,nu=nu,control=list("fnscale"=-1))

################################################################# hard code optimization

require(msm)

n.parameters=2   # total number of parameters
n.iter=100       # total number of iterations
n.particles=20  # total number of particles
pset=array(NA,c(n.iter,n.particles,n.parameters)) # have to add a dimension
lpset=matrix(NA,n.iter,n.particles)
tune=3

for(i in 1:n.particles){
  pset[1,i,1]=runif(1,200,350)
  pset[1,i,2]=runif(1,20,100)
  lpset[1,i]=log.like(pset[1,i,],data,nu=nu)
}

for(t in 2:n.iter){
  for(i in 1:n.particles){
    pstar=rtnorm(n.parameters,pset[t-1,i,],tune,0,Inf)
    lpstar=log.like(pstar, data, nu=nu)
    
    if(lpstar>lpset[t-1,i]){ 
      # if it's better, take it
      pset[t,i,]=pstar
      lpset[t,i]=lpstar
    } else { 
      # if it's worse, reject it
      pset[t,i,]=pset[t-1,i,]
      lpset[t,i]=lpset[t-1,i]
    }
  }
}

# plot it!
par(mfrow=c(1,2))
matplot(pset[,,1],type="l",xlab="Iteration",ylab=expression(mu),lty=1)
abline(h=mu,lwd=3,col=1)

matplot(pset[,,2],type="l",xlab="Iteration",ylab=expression(sigma),lty=1)
abline(h=sigma,lwd=3,col=1)
