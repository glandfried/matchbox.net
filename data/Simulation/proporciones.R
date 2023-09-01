
grilla = seq(0,1,0.01)
alpha = 0.5
beta = 0.5

peliculas = 100
fetures = 5

distribucion = dbeta(grilla, alpha, beta)
plot(grilla,distribucion , type="l", ylim=c(0, 4))



res = matrix(nrow=peliculas, ncol=fetures)
for (i in seq(100)){
    res[i,] = rbeta(5,alpha, beta)
    res[i,] = (res[i,]/sum(res[i,]))*100
}


res

